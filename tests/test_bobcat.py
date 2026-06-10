import math
import pickle

import pytest
import torch

from lambeq.bobcat.lexicon import Atom
from lambeq.bobcat.tagger import extract_topk


def test_fast_int_enum():
    assert pickle.loads(pickle.dumps(Atom('N'))) == Atom('N')


def reference_extract(logits, lengths, top_k, prob_threshold, strategy,
                      skip_index_0):
    """Transcription of the pre-vectorization extraction loop."""
    output_batch = []
    k = min(top_k, logits.size(-1)) if top_k else logits.size(-1)
    logprobs = logits.log_softmax(-1).topk(k)
    for length, sentence_scores, sentence_indices in zip(
            lengths, logprobs.values, logprobs.indices):
        output_list = []
        output_batch.append(output_list)
        for scores, indices in zip(sentence_scores[:length].tolist(),
                                   sentence_indices[:length].tolist()):
            output = []
            output_list.append(output)
            if prob_threshold == 0:
                threshold = -float('inf')
            else:
                top_score = scores[0] if strategy == 'relative' else 0
                threshold = top_score + math.log(prob_threshold)
            for score, index in zip(scores, indices):
                if score < threshold:
                    break
                elif index != 0 or not skip_index_0:
                    output.append((index, score))
    return output_batch


@pytest.mark.parametrize('strategy', ['relative', 'absolute'])
@pytest.mark.parametrize('skip_index_0', [False, True])
@pytest.mark.parametrize('prob_threshold', [0, 0.01, 1])
@pytest.mark.parametrize('top_k', [0, 5])
def test_extract_topk_matches_reference(strategy, skip_index_0,
                                        prob_threshold, top_k):
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 11)
    lengths = [7, 4, 1]
    assert (extract_topk(logits, lengths, top_k, prob_threshold,
                         strategy, skip_index_0)
            == reference_extract(logits, lengths, top_k, prob_threshold,
                                 strategy, skip_index_0))
