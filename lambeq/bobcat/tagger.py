# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CCG tagger model
================

Model for tagging text with CCG tags. This work is based on
the PyTorch BERT model available in Huggingface transformers
(https://huggingface.co/transformers) which is released under
Apache License 2.0.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any, cast, List, Tuple

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import (BertConfig, BertModel, BertPreTrainedModel,
                          PreTrainedModel, PreTrainedTokenizerFast)
from transformers.modeling_outputs import ModelOutput

from lambeq.core.globals import VerbosityLevel

_SpanT = Tuple[int, int]
TagListT = List[Tuple[int, float]]  # list of (index, log probability) tuples

SPAN_MEMO: list[_SpanT] = []


def chart_size(length: int) -> int:
    return length * (length + 1) // 2


def get_chart_spans(length: int) -> list[_SpanT]:
    size = chart_size(length)

    if size <= len(SPAN_MEMO):
        return SPAN_MEMO[:size]

    try:
        last_end = SPAN_MEMO[-1][1]
    except IndexError:
        last_end = -1

    for end in range(last_end + 1, length):
        SPAN_MEMO.extend((start, end) for start in reversed(range(end + 1)))
    return SPAN_MEMO[:]


def idx2span(i: int) -> _SpanT:
    if len(SPAN_MEMO) <= i:
        get_chart_spans(int((2*i) ** 0.5) + 1)
    return SPAN_MEMO[i]


def span2idx(x: int, y: int) -> int:
    return chart_size(y + 1) - x - 1


def extract_topk(logits: torch.Tensor,
                 lengths: Sequence[int],
                 top_k: int,
                 prob_threshold: float,
                 strategy: str,
                 skip_index_0: bool) -> list[list[TagListT]]:
    """Extract the top entries by log probability for each position.

    Parameters
    ----------
    logits : torch.Tensor of shape (batch, positions, classes)
        The raw logits.
    lengths : sequence of int
        The real number of positions per batch entry; positions beyond
        this are padding and are dropped.
    top_k : int
        The maximum number of entries to keep per position. If 0, keep
        all entries.
    prob_threshold : float
        The probability used for the threshold to keep entries.
    strategy : {'relative', 'absolute'}
        If "relative", the probability threshold is relative to the
        highest scoring entry, otherwise it is absolute.
    skip_index_0 : bool
        Whether entries for class index 0 should be dropped.

    Returns
    -------
    list of list of list of tuple of int and float
        For each batch entry, for each position, the kept entries as
        (class index, log probability) tuples.

    """
    logits = logits.float()  # autocast may produce reduced precision
    k = min(top_k, logits.size(-1)) if top_k else logits.size(-1)
    scores, indices = logits.log_softmax(-1).topk(k)

    if prob_threshold == 0:
        mask = torch.ones_like(scores, dtype=torch.bool)
    elif strategy == 'relative':
        mask = scores >= scores[..., :1] + math.log(prob_threshold)
    else:
        mask = scores >= math.log(prob_threshold)
    if skip_index_0:
        mask = mask & (indices != 0)

    # drop padding positions on-device, then transfer only the
    # surviving entries: materialising the full padded
    # (batch, positions, k) tensors as Python objects costs time that
    # grows quadratically with sentence length for the span classifier
    positions = torch.arange(scores.shape[1], device=scores.device)
    length_mask = (positions.unsqueeze(0)
                   < torch.tensor(lengths, device=scores.device)
                          .unsqueeze(1))
    mask = mask & length_mask.unsqueeze(-1)

    # row-major order means entries arrive grouped by (sentence,
    # position), with scores in descending order within each position
    coords = mask.nonzero().tolist()
    kept_scores = scores[mask].tolist()
    kept_indices = indices[mask].tolist()

    output_batch: list[list[TagListT]] = [
        [[] for _ in range(length)] for length in lengths]
    for (sent, pos, _), index, score in zip(coords,
                                            kept_indices,
                                            kept_scores):
        output_batch[sent][pos].append((index, score))
    return output_batch


@dataclass
class ChartClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    tag_logits: torch.FloatTensor | None = None
    span_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class ChartClassifierConfig(BertConfig):
    def __init__(self,
                 empty_span_weight: float | None = None,
                 tags: Sequence[str] = (),
                 cats: Sequence[str] = (),
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tags = list(tags)
        self.cats = list(cats)

        self.empty_span_weight = empty_span_weight

    @property
    def num_cats(self) -> int:
        return len(self.cats)

    @property
    def num_tags(self) -> int:
        return len(self.tags)


class BertForChartClassification(BertPreTrainedModel):
    config_class = ChartClassifierConfig

    def __init__(self, config: ChartClassifierConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)

        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None
                              else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)

        self.tag_classifier = nn.Linear(config.hidden_size, config.num_tags)
        self.span_classifier = nn.Linear(2*config.hidden_size, config.num_cats)

        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        tag_labels: torch.LongTensor | None = None,
        span_labels: torch.LongTensor | None = None,
        word_mask: torch.BoolTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None
    ) -> ChartClassifierOutput | tuple[Any, ...]:
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if word_mask is not None:
            # remove ignored tensors and pack remaining ones
            word_indices = nn.utils.rnn.pad_sequence(
                [sent_word_mask.nonzero().squeeze(dim=-1)
                 for sent_word_mask in word_mask],
                batch_first=True
            )
            word_indices = word_indices.unsqueeze(-1).expand(
                *word_indices.shape, self.config.hidden_size)

            tag_input = sequence_output.gather(-2, word_indices)
        else:
            tag_input = sequence_output

        chart_spans = get_chart_spans(tag_input.shape[-2])
        span_input = tag_input[:, chart_spans].flatten(start_dim=-2)

        tag_logits = self.tag_classifier(self.dropout(tag_input))
        span_logits = self.span_classifier(self.dropout(span_input))

        loss = None
        if (tag_labels is not None
                and span_labels is not None):  # pragma: no cover
            # this is only used for training
            loss_fct_tag = nn.CrossEntropyLoss()
            if self.config.empty_span_weight is not None:
                weight = torch.ones(self.config.num_cats, device=self.device)
                weight[0] = self.config.empty_span_weight
                loss_fct_span = nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fct_span = loss_fct_tag
            pad_value = loss_fct_tag.ignore_index

            tag_padding = tag_labels[:, tag_logits.shape[1]:]
            assert torch.all(tag_padding == pad_value)
            tag_logits = nn.functional.pad(tag_logits,
                                           (0, 0, 0, tag_padding.shape[-1]),
                                           value=pad_value)
            tag_loss = loss_fct_tag(tag_logits.view(-1, self.config.num_tags),
                                    tag_labels.view(-1))

            span_padding = span_labels[:, span_logits.shape[1]:]
            assert torch.all(span_padding == pad_value)
            span_logits = nn.functional.pad(span_logits,
                                            (0, 0, 0, span_padding.shape[-1]),
                                            value=pad_value)
            span_loss = loss_fct_span(
                    span_logits.view(-1, self.config.num_cats),
                    span_labels.view(-1))

            n_tags = tag_labels[tag_labels != pad_value].count_nonzero()
            n_spans = span_labels[span_labels != pad_value].count_nonzero()

            loss = (tag_loss*n_tags + span_loss*n_spans) / (n_tags + n_spans)

        if not return_dict:  # pragma: no cover
            output = (tag_logits, span_logits, *outputs[2:])
            return (loss, *output) if loss is not None else output

        return ChartClassifierOutput(loss=loss,
                                     tag_logits=tag_logits,
                                     span_logits=span_logits,
                                     hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)


@dataclass
class TaggerOutputSentence:
    """A sentence in the tagger output.

    Parameters
    ----------
    words : list of str
        The tokens in the sentence.
    tags : list of list of tuple of int and float
        A list of tag indices with their log probability for each word.
    spans : list of tuple
        A list of tuples of:
            - an integer denoting the start of the span
            - an integer denoting the end of the span
            - a list of cat indices with their log probability

    """
    words: list[str]
    tags: list[TagListT]
    spans: list[tuple[int, int, TagListT]]


@dataclass
class TaggerOutput:
    tags: list[str]
    cats: list[str]
    sentences: list[TaggerOutputSentence]

    def asdict(self) -> dict[str, Any]:  # pragma: no cover
        return asdict(self)

    @staticmethod
    def tags_str(tags: list[tuple[int, float]],
                 precision: int) -> str:  # pragma: no cover
        return ','.join(f'{idx}={logp:.{precision}}' for idx, logp in tags)

    def astext(self, precision: int = 3) -> str:  # pragma: no cover
        """Convert into a form that can be passed to Java C&C."""

        tags = ' '.join(self.tags)
        cats = ' '.join(self.cats)

        output = f'tags: {tags}\ncats: {cats}\nsentences:'
        for sentence in self.sentences:
            words = ' '.join(sentence.words)
            tags = ' '.join(self.tags_str(tag_list, precision)
                            for tag_list in sentence.tags)
            spans = ' '.join(f'{start},{end}:' + self.tags_str(tags, precision)
                             for start, end, tags in sentence.spans)

            output += f'\n words: {words}\n tags: {tags}\n spans: {spans}\n'
        return output


class Tagger:
    # TODO can this be made into a huggingface Pipeline in a good way?

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerFast,
                 batch_size: int = 1,
                 tag_top_k: int = 1,
                 tag_prob_threshold: float = 1,
                 tag_prob_threshold_strategy: str = 'relative',
                 span_top_k: int = 1,
                 span_prob_threshold: float = 1,
                 span_prob_threshold_strategy: str = 'relative',
                 max_spans_per_batch: int | None = None,
                 dtype: str | None = None) -> None:
        strategies = ('absolute', 'relative')

        if not (batch_size >= 1 and batch_size == int(batch_size)):
            raise ValueError(f'Invalid `batch_size`: {batch_size}')
        if not (tag_top_k >= 0 and tag_top_k == int(tag_top_k)):
            raise ValueError(f'Invalid `tag_top_k`: {tag_top_k}')
        if not 0 <= tag_prob_threshold <= 1:
            raise ValueError('Invalid `tag_prob_threshold`: '
                             f'{tag_prob_threshold}')
        if tag_prob_threshold_strategy not in strategies:
            raise ValueError('Invalid `tag_prob_threshold_strategy`: '
                             f'{tag_prob_threshold_strategy}')

        if not (span_top_k >= 0 and span_top_k == int(span_top_k)):
            raise ValueError(f'Invalid `span_top_k`: {span_top_k}')
        if not 0 <= span_prob_threshold <= 1:
            raise ValueError('Invalid `span_prob_threshold`: '
                             f'{span_prob_threshold}')
        if span_prob_threshold_strategy not in strategies:
            raise ValueError('Invalid `span_prob_threshold_strategy`: '
                             f'{span_prob_threshold_strategy}')

        if max_spans_per_batch is not None and not (
                max_spans_per_batch >= 1
                and max_spans_per_batch == int(max_spans_per_batch)):
            raise ValueError('Invalid `max_spans_per_batch`: '
                             f'{max_spans_per_batch}')

        if dtype is not None and dtype not in ('float16', 'bfloat16'):
            raise ValueError(f'Invalid `dtype`: {dtype}')

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.tag_top_k = int(tag_top_k)
        self.tag_prob_threshold = tag_prob_threshold
        self.tag_prob_threshold_strategy = tag_prob_threshold_strategy
        self.span_top_k = int(span_top_k)
        self.span_prob_threshold = span_prob_threshold
        self.span_prob_threshold_strategy = span_prob_threshold_strategy
        self.max_spans_per_batch = (None if max_spans_per_batch is None
                                    else int(max_spans_per_batch))
        self.dtype = dtype

    def prepare_inputs(self,
                       inputs: Sequence[Sequence[str]],
                       word_mask: bool = False) -> dict[str, Any]:
        """Prepare a batch of sentences for parsing."""
        encodings: dict[str, Any]
        encodings = self.tokenizer(inputs,
                                   is_split_into_words=True,
                                   return_offsets_mapping=word_mask,
                                   padding=True,
                                   truncation=True)

        if word_mask:
            offset_mapping = encodings.pop('offset_mapping')
            encodings['word_mask'] = [
                    [not start and end for start, end in offsets]
                    for offsets in offset_mapping]

        return encodings

    @torch.inference_mode()
    def parse(self,
              inputs: Sequence[Sequence[str]]) -> list[TaggerOutputSentence]:
        """Parse a batch of sentences."""
        encodings = self.prepare_inputs(inputs, word_mask=True)
        if self.dtype is None:
            autocast = contextlib.nullcontext()
        else:
            autocast = torch.autocast(device_type=self.model.device.type,
                                      dtype=getattr(torch, self.dtype))
        with autocast:
            outputs = self.model(
                **{k: torch.as_tensor(v, device=self.model.device)
                   for k, v in encodings.items()})

        tag_lengths = [len(sentence) for sentence in inputs]
        span_lengths = [chart_size(length) for length in tag_lengths]

        tag_output = extract_topk(outputs.tag_logits,
                                  tag_lengths,
                                  self.tag_top_k,
                                  self.tag_prob_threshold,
                                  self.tag_prob_threshold_strategy,
                                  skip_index_0=False)
        span_output = extract_topk(outputs.span_logits,
                                   span_lengths,
                                   self.span_top_k,
                                   self.span_prob_threshold,
                                   self.span_prob_threshold_strategy,
                                   skip_index_0=True)

        spans_list = [[(*idx2span(i), output)
                       for i, output in enumerate(sent_span_output)
                       if output]
                      for sent_span_output in span_output]

        return [TaggerOutputSentence(list(words), tags, spans)
                for words, tags, spans in zip(inputs, tag_output, spans_list)]

    def make_batches(self,
                     inputs: Sequence[Sequence[str]],
                     batch_size: int) -> list[list[int]]:
        """Group sentence indices into length-sorted batches.

        Batching sentences of similar length together avoids wasting
        compute on padding, which is especially costly for the span
        classifier whose size grows quadratically with sentence length.

        If `max_spans_per_batch` is set, it overrides `batch_size`:
        each batch takes as many sentences as fit within that padded
        span count, capping the memory used per batch. A sentence that
        exceeds the budget on its own forms a singleton batch, which
        may exceed the cap.

        """
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
        if self.max_spans_per_batch is None:
            return [order[i:i + batch_size]
                    for i in range(0, len(order), batch_size)]

        batches: list[list[int]] = []
        batch: list[int] = []
        for i in order:
            # `order` is sorted, so sentence `i` is the longest in the
            # batch and determines its padded length
            padded_spans = (len(batch) + 1) * chart_size(len(inputs[i]))
            if batch and padded_spans > self.max_spans_per_batch:
                batches.append(batch)
                batch = []
            batch.append(i)
        if batch:
            batches.append(batch)
        return batches

    def __call__(self,
                 inputs: Sequence[Sequence[str]],
                 batch_size: int | None = None,
                 verbose: str = VerbosityLevel.PROGRESS.value) -> TaggerOutput:
        """Parse a list of sentences."""
        if batch_size is None:
            batch_size = self.batch_size
        elif batch_size < 1:
            raise ValueError(f'Invalid `batch_size`: {batch_size}')

        sentences: list[TaggerOutputSentence | None] = [None] * len(inputs)
        for batch in tqdm(
                self.make_batches(inputs, batch_size),
                desc='Tagging sentences',
                leave=False,
                disable=verbose != VerbosityLevel.PROGRESS.value):
            results = self.parse([inputs[i] for i in batch])
            for i, sentence in zip(batch, results):
                sentences[i] = sentence

        return TaggerOutput(
                tags=self.model.config.tags,
                cats=self.model.config.cats,
                sentences=cast(List[TaggerOutputSentence], sentences))
