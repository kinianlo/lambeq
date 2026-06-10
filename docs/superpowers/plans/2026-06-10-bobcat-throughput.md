# BobcatParser Throughput Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Higher sentences/sec and lower peak memory for `BobcatParser` on CPU and GPU, with all changes upstream-mergeable.

**Architecture:** `BobcatParser` = BERT supertagger (`lambeq/bobcat/tagger.py`) + pure-Python CKY chart parser (`lambeq/bobcat/parser.py`), glued in `lambeq/text2diagram/model_based_reader/bobcat_parser.py`. We add a benchmark harness, then six independent optimizations to those two stages: inference_mode, length-sorted batching, span-budget batching, vectorized post-processing, opt-in autocast dtype, and opt-in multiprocess chart parsing.

**Tech Stack:** Python, PyTorch, transformers, multiprocessing, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-bobcat-throughput-design.md`

---

## Conventions used by every task

- Interpreter: `PY=~/.pyenv/versions/qnlp/bin/python` (pyenv env `qnlp`,
  Python 3.11, torch 2.5.0). Run everything from the repo root
  `/home/kinianlo/projects/lambeq` on branch `bobcat-throughput`.
- The Bobcat model (~1.4 GB) is already cached at
  `~/.cache/lambeq/bobcat/bobcat`, so tests do not download anything.
- Model-dependent tests live in
  `tests/text2diagram/model_based_reader/test_bobcat_parser.py` (module-scoped
  `bobcat_parser` fixture already loads the model once). Pure-tensor tests live
  in `tests/test_bobcat.py` (no model load).
- Tests that mutate the shared `bobcat_parser` fixture MUST restore the
  attribute in a `finally` block (existing tests follow this pattern).
- Benchmark results are appended to `benchmarks/RESULTS.md` after each task,
  under a heading naming the task. Benchmark command (CPU, 200 sentences):
  `$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200`
  and additionally with `--device cuda` if Task 0 found CUDA available.
- Commit messages: plain sentence style (matches repo history), ending with
  the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 0: Environment setup and baseline

**Files:** none created (environment only)

- [ ] **Step 1: Regenerate `lambeq/version.py` and install editable**

The repo copy of lambeq currently shadows the installed package but lacks the
setuptools_scm-generated `lambeq/version.py`, so `import lambeq` fails from
the repo root. Fix:

```bash
PY=~/.pyenv/versions/qnlp/bin/python
cd /home/kinianlo/projects/lambeq
$PY -m pip install -e . --no-deps
$PY -c "import lambeq; print(lambeq.__version__, lambeq.__file__)"
```

Expected: prints a version and a path inside `/home/kinianlo/projects/lambeq`.

- [ ] **Step 2: Check CUDA availability**

```bash
$PY -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Record the answer. If `False`, all `--device cuda` benchmark steps in later
tasks are skipped (CPU numbers still collected; GPU numbers can be gathered
later on a GPU machine).

- [ ] **Step 3: Run the existing Bobcat test suite as a green baseline**

```bash
$PY -m pytest tests/test_bobcat.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all tests PASS. Do not proceed until they do.

---

### Task 1: Benchmark harness

**Files:**
- Create: `benchmarks/bobcat_throughput.py`
- Create: `benchmarks/RESULTS.md`

- [ ] **Step 1: Generate a benchmark sentence file (not committed)**

```bash
$PY - <<'EOF'
import itertools
subjects = ['Alice', 'Bob', 'the cat', 'my neighbour', 'the old man']
verbs = ['likes', 'sees', 'helps', 'admires', 'follows']
objects = ['Bob', 'the dog', 'a book about science', 'the red car', 'music']
adjuncts = ['', 'in the park', 'on a rainy day', 'with great enthusiasm',
            'before breakfast', 'while reading a long and boring report']
with open('/tmp/bobcat_bench.txt', 'w') as f:
    for s, v, o, a in itertools.product(subjects, verbs, objects, adjuncts):
        f.write(f'{s} {v} {o} {a}'.strip() + '\n')
EOF
wc -l /tmp/bobcat_bench.txt
```

Expected: `750 /tmp/bobcat_bench.txt`.

- [ ] **Step 2: Write the benchmark script**

Create `benchmarks/bobcat_throughput.py` with exactly this content:

```python
#!/usr/bin/env python3
"""Benchmark BobcatParser throughput and memory.

Reads a text file with one (whitespace-tokenisable) sentence per line and
reports tagging time, chart-parsing time, sentences/sec and peak memory.

Example:
    python benchmarks/bobcat_throughput.py sentences.txt --device cuda
"""
import argparse
import resource
import time

import torch

from lambeq import BobcatParser, VerbosityLevel


def peak_rss_mb() -> float:
    """Peak resident set size of this process and its children, in MB."""
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (self_kb + children_kb) / 1024


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('sentence_file',
                      help='text file with one sentence per line')
    argp.add_argument('--device', default='cpu')
    argp.add_argument('--num', type=int, default=None,
                      help='use only the first NUM sentences')
    argp.add_argument('--batch-size', type=int, default=None)
    argp.add_argument('--max-spans-per-batch', type=int, default=None)
    argp.add_argument('--dtype', default=None,
                      help="e.g. 'float16' or 'bfloat16'")
    argp.add_argument('--n-jobs', type=int, default=1)
    args = argp.parse_args()

    with open(args.sentence_file) as f:
        sentences = [line.split() for line in f if line.strip()]
    sentences = sentences[:args.num]

    kwargs = {}
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    if args.max_spans_per_batch is not None:
        kwargs['max_spans_per_batch'] = args.max_spans_per_batch
    if args.dtype is not None:
        kwargs['dtype'] = args.dtype

    parser = BobcatParser(device=args.device,
                          verbose=VerbosityLevel.SUPPRESS.value,
                          **kwargs)

    parse_kwargs = {}
    if args.n_jobs != 1:
        parse_kwargs['n_jobs'] = args.n_jobs

    cuda = torch.device(args.device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    # warm-up: initialise lazy state (e.g. CUDA kernels)
    parser.sentences2trees([['Alice', 'likes', 'Bob']], tokenised=True)

    # Stage timing: the tagger is timed standalone, then the full
    # pipeline is timed; chart-parse time is the difference (the tagger
    # therefore runs twice, which is accepted for simplicity).
    start = time.perf_counter()
    parser.tagger(sentences, verbose=VerbosityLevel.SUPPRESS.value)
    if cuda:
        torch.cuda.synchronize()
    tag_time = time.perf_counter() - start

    start = time.perf_counter()
    trees = parser.sentences2trees(sentences,
                                   tokenised=True,
                                   suppress_exceptions=True,
                                   **parse_kwargs)
    pipeline_time = time.perf_counter() - start

    parse_time = pipeline_time - tag_time
    n = len(sentences)
    failures = sum(tree is None for tree in trees)

    print(f'config:           {kwargs} {parse_kwargs} '
          f'device={args.device}')
    print(f'sentences:        {n} ({failures} failed)')
    print(f'tagging:          {tag_time:.2f}s ({n / tag_time:.1f} sent/s)')
    print(f'chart parsing:    {parse_time:.2f}s ({n / parse_time:.1f} sent/s)')
    print(f'end-to-end:       {pipeline_time:.2f}s '
          f'({n / pipeline_time:.1f} sent/s)')
    print(f'peak RSS:         {peak_rss_mb():.0f} MB')
    if cuda:
        print(f'peak CUDA memory: '
              f'{torch.cuda.max_memory_allocated() / 2**20:.0f} MB')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Smoke-run the benchmark**

```bash
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 20
```

Expected: a report like the `print` lines above, 0 failed, no traceback.

- [ ] **Step 4: Record the baseline**

Create `benchmarks/RESULTS.md`:

```markdown
# BobcatParser throughput benchmark results

Corpus: 750 generated sentences (3-12 words), `/tmp/bobcat_bench.txt`,
see Task 1 of `docs/superpowers/plans/2026-06-10-bobcat-throughput.md`.
Machine: <fill in: CPU model, GPU model if any, RAM>.

## Baseline (before optimizations)
```

Then run and paste the full output under the heading:

```bash
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200
# and, if CUDA available:
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200 --device cuda
```

Fill in the machine line (`lscpu | grep 'Model name'`, `nvidia-smi -L`).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/bobcat_throughput.py benchmarks/RESULTS.md
git commit -m "Add throughput benchmark for BobcatParser

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `torch.inference_mode()` in `Tagger.parse`

**Files:**
- Modify: `lambeq/bobcat/tagger.py` (method `Tagger.parse`, ~line 338)
- Test: `tests/text2diagram/model_based_reader/test_bobcat_parser.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/text2diagram/model_based_reader/test_bobcat_parser.py`
(add `import torch` to the imports at the top):

```python
def test_tagger_uses_inference_mode(bobcat_parser):
    recorded = []
    handle = bobcat_parser.tagger.model.register_forward_pre_hook(
        lambda module, args: recorded.append(
            torch.is_inference_mode_enabled()))
    try:
        bobcat_parser.tagger.parse([['Alice', 'likes', 'Bob']])
    finally:
        handle.remove()
    assert recorded == [True]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
$PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_tagger_uses_inference_mode -q
```

Expected: FAIL with `assert [False] == [True]`.

- [ ] **Step 3: Implement**

In `lambeq/bobcat/tagger.py`, decorate `Tagger.parse`:

```python
    @torch.inference_mode()
    def parse(self,
              inputs: Sequence[Sequence[str]]) -> list[TaggerOutputSentence]:
        """Parse a batch of sentences."""
```

(The whole method body, including post-processing, now runs under
inference mode; nothing in it needs gradients.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
$PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all PASS.

- [ ] **Step 5: Benchmark and record**

Run the benchmark commands from "Conventions" and append the output to
`benchmarks/RESULTS.md` under `## After inference_mode (Task 2)`.

- [ ] **Step 6: Commit**

```bash
git add lambeq/bobcat/tagger.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py \
    benchmarks/RESULTS.md
git commit -m "Run Bobcat tagger under torch.inference_mode

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Length-sorted batching in `Tagger.__call__`

**Files:**
- Modify: `lambeq/bobcat/tagger.py` (imports; method `Tagger.__call__`, ~line 395)
- Test: `tests/text2diagram/model_based_reader/test_bobcat_parser.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/text2diagram/model_based_reader/test_bobcat_parser.py`:

```python
def test_make_batches_sorts_by_length(bobcat_parser):
    sentences = [['a'] * n for n in (5, 1, 3, 2, 4)]
    batches = bobcat_parser.tagger.make_batches(sentences, batch_size=2)
    assert batches == [[1, 3], [2, 4], [0]]


def test_tagger_restores_input_order(bobcat_parser):
    sentences = [
        'Alice likes Bob and Claire likes Dave'.split(),
        'Alice likes Bob'.split(),
        'I do'.split(),
        'What Alice is and is not .'.split(),
    ]
    output = bobcat_parser.tagger(sentences, batch_size=2,
                                  verbose=VerbosityLevel.SUPPRESS.value)
    assert [s.words for s in output.sentences] == sentences
```

- [ ] **Step 2: Run tests to verify the new one fails**

```bash
$PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_make_batches_sorts_by_length \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_tagger_restores_input_order -q
```

Expected: `test_make_batches_sorts_by_length` FAILS with
`AttributeError: 'Tagger' object has no attribute 'make_batches'`
(`test_tagger_restores_input_order` may already pass — it is the regression
guard for the reordering being introduced).

- [ ] **Step 3: Implement**

In `lambeq/bobcat/tagger.py`:

1. Change the tqdm import (line 33) from
   `from tqdm.auto import trange` to `from tqdm.auto import tqdm`.
2. Change the typing import (line 29) to
   `from typing import Any, cast, List, Tuple`.
3. Add a method to `Tagger` (before `__call__`):

```python
    def make_batches(self,
                     inputs: Sequence[Sequence[str]],
                     batch_size: int) -> list[list[int]]:
        """Group sentence indices into length-sorted batches.

        Batching sentences of similar length together avoids wasting
        compute on padding, which is especially costly for the span
        classifier whose size grows quadratically with sentence length.

        """
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
        return [order[i:i + batch_size]
                for i in range(0, len(order), batch_size)]
```

4. Replace the whole `__call__` method with:

```python
    def __call__(self,
                 inputs: Sequence[Sequence[str]],
                 batch_size: int | None = None,
                 verbose: str = VerbosityLevel.PROGRESS.value) -> TaggerOutput:
        """Parse a list of sentences."""
        if batch_size is None:
            batch_size = self.batch_size

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
```

- [ ] **Step 4: Run the Bobcat test files**

```bash
$PY -m pytest tests/test_bobcat.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all PASS.

- [ ] **Step 5: Benchmark and record**

Append to `benchmarks/RESULTS.md` under `## After length-sorted batching
(Task 3)`. Also try a larger batch size for comparison:

```bash
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200 --batch-size 32
```

- [ ] **Step 6: Commit**

```bash
git add lambeq/bobcat/tagger.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py \
    benchmarks/RESULTS.md
git commit -m "Sort sentences by length before batching in Bobcat tagger

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Span-budget batching (`max_spans_per_batch`)

**Files:**
- Modify: `lambeq/bobcat/tagger.py` (`Tagger.__init__`, `Tagger.make_batches`)
- Modify: `lambeq/text2diagram/model_based_reader/bobcat_parser.py`
  (`_initialise_model`, ~line 167; class docstring "Other Parameters")
- Test: `tests/text2diagram/model_based_reader/test_bobcat_parser.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/text2diagram/model_based_reader/test_bobcat_parser.py`
(extend the top-of-file imports with
`from lambeq.bobcat.tagger import Tagger, chart_size`):

```python
def test_make_batches_respects_span_budget(bobcat_parser):
    tagger = bobcat_parser.tagger
    tagger.max_spans_per_batch = 20
    try:
        sentences = [['a'] * n for n in (1, 2, 3, 4, 5)]
        batches = tagger.make_batches(sentences, batch_size=1000)
        # every sentence appears exactly once
        assert sorted(i for batch in batches for i in batch) == [0, 1, 2, 3, 4]
        for batch in batches:
            max_len = max(len(sentences[i]) for i in batch)
            padded_spans = len(batch) * chart_size(max_len)
            assert len(batch) == 1 or padded_spans <= 20
    finally:
        tagger.max_spans_per_batch = None


def test_invalid_max_spans_per_batch(bobcat_parser):
    with pytest.raises(ValueError):
        Tagger(bobcat_parser.tagger.model,
               bobcat_parser.tagger.tokenizer,
               max_spans_per_batch=0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
$PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_make_batches_respects_span_budget \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_invalid_max_spans_per_batch -q
```

Expected: first FAILS (budget ignored: one batch of 5 sentences,
5 × 15 = 75 > 20), second FAILS (`TypeError: unexpected keyword argument`).

- [ ] **Step 3: Implement in `Tagger`**

In `lambeq/bobcat/tagger.py`:

1. Add the parameter to `Tagger.__init__` (after
   `span_prob_threshold_strategy: str = 'relative'`):

```python
                 max_spans_per_batch: int | None = None) -> None:
```

2. Add validation (after the `span_prob_threshold_strategy` check) and the
   attribute (next to `self.span_prob_threshold_strategy = ...`):

```python
        if max_spans_per_batch is not None and not (
                max_spans_per_batch >= 1
                and max_spans_per_batch == int(max_spans_per_batch)):
            raise ValueError('Invalid `max_spans_per_batch`: '
                             f'{max_spans_per_batch}')
```

```python
        self.max_spans_per_batch = (None if max_spans_per_batch is None
                                    else int(max_spans_per_batch))
```

3. Replace `make_batches` with:

```python
    def make_batches(self,
                     inputs: Sequence[Sequence[str]],
                     batch_size: int) -> list[list[int]]:
        """Group sentence indices into length-sorted batches.

        Batching sentences of similar length together avoids wasting
        compute on padding, which is especially costly for the span
        classifier whose size grows quadratically with sentence length.

        If `max_spans_per_batch` is set, it overrides `batch_size`:
        each batch takes as many sentences as fit within that padded
        span count, so memory use stays flat across batches. A sentence
        that exceeds the budget on its own forms a singleton batch.

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
```

- [ ] **Step 4: Plumb through `BobcatParser`**

The shipped `pipeline_config.json` does not contain new keys, so
`_initialise_model` would reject them. In
`lambeq/text2diagram/model_based_reader/bobcat_parser.py`, inside
`_initialise_model`, insert between the existing config-override loop and the
`if kwargs:` check:

```python
        # parameters that postdate the shipped pipeline_config.json
        for key in ('max_spans_per_batch',):
            if key in kwargs:
                config['tagger'][key] = kwargs.pop(key)
```

Also add to the "Other Parameters / Tagger parameters" docstring section of
`BobcatParser.__init__` (after the `span_prob_threshold_strategy` entry):

```python
        max_spans_per_batch : int, optional
            If set, overrides `batch_size`: each batch contains as many
            sentences as fit in this padded span count, keeping memory
            usage flat regardless of sentence length.
```

- [ ] **Step 5: Run the tests**

```bash
$PY -m pytest tests/test_bobcat.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all PASS.

- [ ] **Step 6: Benchmark and record**

Append under `## After span-budget batching (Task 4)`:

```bash
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200 --max-spans-per-batch 5000
# if CUDA available, also:
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200 --device cuda --max-spans-per-batch 20000
```

(Try a couple of budgets; note the peak-memory column especially.)

- [ ] **Step 7: Commit**

```bash
git add lambeq/bobcat/tagger.py \
    lambeq/text2diagram/model_based_reader/bobcat_parser.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py \
    benchmarks/RESULTS.md
git commit -m "Add span-budget batching to Bobcat tagger

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Vectorized tag/span post-processing

**Files:**
- Modify: `lambeq/bobcat/tagger.py` (new module function `extract_topk`;
  rewrite of `Tagger.parse` body)
- Test: `tests/test_bobcat.py` (pure-tensor, no model needed)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bobcat.py`:

```python
import math

import pytest
import torch

from lambeq.bobcat.tagger import extract_topk


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
$PY -m pytest tests/test_bobcat.py -q
```

Expected: FAIL with `ImportError: cannot import name 'extract_topk'`.

- [ ] **Step 3: Implement `extract_topk` and rewrite `Tagger.parse`**

In `lambeq/bobcat/tagger.py`, add a module-level function after
`span2idx` (~line 74):

```python
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

    output_batch = []
    for length, sent_scores, sent_indices, sent_mask in zip(
            lengths, scores.tolist(), indices.tolist(), mask.tolist()):
        output_batch.append(
            [[(index, score)
              for score, index, keep
              in zip(pos_scores, pos_indices, pos_mask) if keep]
             for pos_scores, pos_indices, pos_mask
             in zip(sent_scores[:length],
                    sent_indices[:length],
                    sent_mask[:length])])
    return output_batch
```

Then replace the body of `Tagger.parse` from `tag_output: ...` down to (but
not including) the `spans_list = ...` statement with:

```python
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
```

(The `spans_list` comprehension and the final `return` stay unchanged. The
old `tag_args`/`span_args` tuples and the big `for output_batch, ...` loop
are deleted.)

- [ ] **Step 4: Run the tests**

```bash
$PY -m pytest tests/test_bobcat.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all PASS (the end-to-end diagram tests confirm the rewiring of
`parse` is faithful).

- [ ] **Step 5: Benchmark and record**

Append under `## After vectorized post-processing (Task 5)`.

- [ ] **Step 6: Commit**

```bash
git add lambeq/bobcat/tagger.py tests/test_bobcat.py benchmarks/RESULTS.md
git commit -m "Vectorize tag and span post-processing in Bobcat tagger

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Opt-in reduced precision (`dtype`)

**Files:**
- Modify: `lambeq/bobcat/tagger.py` (`Tagger.__init__`, `Tagger.parse`)
- Modify: `lambeq/text2diagram/model_based_reader/bobcat_parser.py`
  (extra-keys tuple, docstring)
- Test: `tests/text2diagram/model_based_reader/test_bobcat_parser.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/text2diagram/model_based_reader/test_bobcat_parser.py`:

```python
def test_invalid_dtype(bobcat_parser):
    with pytest.raises(ValueError):
        Tagger(bobcat_parser.tagger.model,
               bobcat_parser.tagger.tokenizer,
               dtype='float8')


def test_reduced_precision_tagging(bobcat_parser, sentence):
    tagger = bobcat_parser.tagger
    assert tagger.dtype is None
    tagger.dtype = 'bfloat16'
    try:
        assert bobcat_parser.sentence2tree(sentence) is not None
    finally:
        tagger.dtype = None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
$PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_invalid_dtype \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_reduced_precision_tagging -q
```

Expected: both FAIL (`TypeError: unexpected keyword argument 'dtype'` /
`AttributeError: 'Tagger' object has no attribute 'dtype'`).

- [ ] **Step 3: Implement**

In `lambeq/bobcat/tagger.py`:

1. Add `import contextlib` to the imports (stdlib group, before
   `from collections.abc import Sequence`).
2. Extend `Tagger.__init__` signature (after `max_spans_per_batch`):

```python
                 dtype: str | None = None) -> None:
```

3. Add validation (after the `max_spans_per_batch` check) and attribute:

```python
        if dtype is not None and dtype not in ('float16', 'bfloat16'):
            raise ValueError(f'Invalid `dtype`: {dtype}')
```

```python
        self.dtype = dtype
```

4. In `Tagger.parse`, wrap the model call:

Replace

```python
        outputs = self.model(**{k: torch.as_tensor(v, device=self.model.device)
                                for k, v in encodings.items()})
```

with

```python
        if self.dtype is None:
            autocast = contextlib.nullcontext()
        else:
            autocast = torch.autocast(device_type=self.model.device.type,
                                      dtype=getattr(torch, self.dtype))
        with autocast:
            outputs = self.model(
                **{k: torch.as_tensor(v, device=self.model.device)
                   for k, v in encodings.items()})
```

(`extract_topk` already casts logits with `.float()`, so post-processing
precision is unaffected.)

5. In `lambeq/text2diagram/model_based_reader/bobcat_parser.py`, extend the
   extra-keys tuple added in Task 4:

```python
        for key in ('max_spans_per_batch', 'dtype'):
```

and add to the docstring's tagger-parameters section:

```python
        dtype : str, optional
            If set (e.g. `'float16'` or `'bfloat16'`), the tagger
            forward pass runs under `torch.autocast` with this dtype,
            trading a little numerical precision for speed and memory.
            Use `'bfloat16'` on CPU; on CUDA both usually work.
```

- [ ] **Step 4: Run the tests**

```bash
$PY -m pytest tests/test_bobcat.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all PASS.

- [ ] **Step 5: Benchmark and record**

Append under `## After opt-in dtype (Task 6)`:

```bash
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200 --dtype bfloat16
# if CUDA available:
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200 --device cuda --dtype float16
```

Also note the failure count: it should stay 0 (mild deviations acceptable,
parse failures not expected on this simple corpus).

- [ ] **Step 6: Commit**

```bash
git add lambeq/bobcat/tagger.py \
    lambeq/text2diagram/model_based_reader/bobcat_parser.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py \
    benchmarks/RESULTS.md
git commit -m "Add opt-in reduced-precision inference to Bobcat tagger

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Opt-in parallel chart parsing (`n_jobs`)

**Files:**
- Modify: `lambeq/text2diagram/model_based_reader/bobcat_parser.py`
  (imports, module-level worker functions, `sentences2trees`)
- Test: `tests/text2diagram/model_based_reader/test_bobcat_parser.py`

- [ ] **Step 1: Verify pickling preconditions**

Workers return `CCGTree`s (whose `metadata['original']` holds a Bobcat
`ParseTree`) and, on spawn platforms, receive a `ChartParser` via initargs.
Verify both pickle:

```bash
$PY - <<'EOF'
import pickle
from lambeq import BobcatParser, VerbosityLevel
p = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value)
tree = p.sentence2tree('What Alice is and is not .')
data = pickle.dumps(tree)
assert pickle.loads(data) == tree
print('CCGTree pickles:', len(data), 'bytes')
print('ChartParser pickles:', len(pickle.dumps(p.parser)), 'bytes')
EOF
```

Expected: both lines print. **If either raises, STOP this task and report
back — the design decision (e.g. stripping `metadata['original']` in worker
results) must go back to the user.**

- [ ] **Step 2: Write the failing tests**

Add to `tests/text2diagram/model_based_reader/test_bobcat_parser.py`:

```python
def test_parallel_parsing_matches_serial(bobcat_parser):
    sentences = ['Alice likes Bob',
                 'What Alice is and is not .',
                 'I do not like Bob']
    serial = bobcat_parser.sentences2trees(
        sentences, verbose=VerbosityLevel.SUPPRESS.value)
    parallel = bobcat_parser.sentences2trees(
        sentences, n_jobs=2, verbose=VerbosityLevel.SUPPRESS.value)
    assert parallel == serial


def test_invalid_n_jobs(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        bobcat_parser.sentences2trees([sentence], n_jobs=0)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
$PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_parallel_parsing_matches_serial \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_invalid_n_jobs -q
```

Expected: both FAIL with
`TypeError: ... unexpected keyword argument 'n_jobs'`.

- [ ] **Step 4: Implement**

In `lambeq/text2diagram/model_based_reader/bobcat_parser.py`:

1. Add imports `import multiprocessing` and `import os` (stdlib group), and
   add `ChartParser` usage is already imported via the existing
   `from lambeq.bobcat import (...)` line.

2. Add module-level worker state and functions after the `BobcatParseError`
   class (workers must be importable, so they cannot be closures or methods):

```python
_worker_parser: ChartParser | None = None
_worker_tags: list[str] | None = None
_worker_suppress_exceptions = False


def _init_worker(parser: ChartParser,
                 tags: list[str],
                 suppress_exceptions: bool) -> None:
    global _worker_parser, _worker_tags, _worker_suppress_exceptions
    _worker_parser = parser
    _worker_tags = tags
    _worker_suppress_exceptions = suppress_exceptions


def _parse_tagged_sentence(sent: TaggerOutputSentence) -> CCGTree | None:
    assert _worker_parser is not None and _worker_tags is not None
    try:
        sentence_input = BobcatParser._prepare_sentence(sent, _worker_tags)
        result = _worker_parser(sentence_input)
        return BobcatParser._build_ccgtree(result[0])
    except Exception as e:
        if _worker_suppress_exceptions:
            return None
        raise BobcatParseError(' '.join(sent.words)) from e
```

3. Extend the `sentences2trees` signature:

```python
    def sentences2trees(
        self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None,
        n_jobs: int = 1
    ) -> list[CCGTree] | None:
```

and document it in the method docstring after `verbose`:

```python
        n_jobs : int, default: 1
            The number of processes used for chart parsing the tagged
            sentences. Use -1 for all available cores. The tagger stage
            is unaffected.
```

4. Add validation next to the existing `verbose` validation:

```python
        if not (n_jobs == -1 or n_jobs >= 1):
            raise ValueError(f'Invalid `n_jobs`: {n_jobs}')
```

5. Inside `if sentences_valid:`, keep the tagging part unchanged, then wrap
the existing serial `for sent in tqdm(...)` loop in `if n_jobs == 1:` and add
the parallel branch:

```python
            if n_jobs == 1:
                for sent in tqdm(
                        tag_results.sentences,
                        desc='Parsing tagged sentences',
                        leave=False,
                        disable=verbose != VerbosityLevel.PROGRESS.value):

                    try:
                        sentence_input = self._prepare_sentence(sent, tags)
                        result = self.parser(sentence_input)
                        trees.append(self._build_ccgtree(result[0]))
                    except Exception as e:
                        if suppress_exceptions:
                            trees.append(None)
                        else:
                            raise BobcatParseError(
                                ' '.join(sent.words)) from e
            else:
                processes = os.cpu_count() if n_jobs == -1 else n_jobs
                with multiprocessing.Pool(
                        processes,
                        initializer=_init_worker,
                        initargs=(self.parser,
                                  tags,
                                  suppress_exceptions)) as pool:
                    trees = list(tqdm(
                        pool.imap(_parse_tagged_sentence,
                                  tag_results.sentences),
                        desc='Parsing tagged sentences',
                        total=len(tag_results.sentences),
                        leave=False,
                        disable=verbose != VerbosityLevel.PROGRESS.value))
```

(On fork platforms the workers inherit `self.parser` for free; on spawn
platforms the initargs are pickled — covered by Step 1. A failing sentence
raises `BobcatParseError` inside the worker, which `imap` re-raises in the
parent, preserving the `suppress_exceptions=False` behaviour.)

- [ ] **Step 5: Run the tests**

```bash
$PY -m pytest tests/test_bobcat.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all PASS.

- [ ] **Step 6: Benchmark and record**

Append under `## After parallel chart parsing (Task 7)`:

```bash
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200 --n-jobs 4
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200 --n-jobs -1
# if CUDA available (tagger fast, chart parser is then the wall):
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --num 200 --device cuda --n-jobs -1
```

- [ ] **Step 7: Commit**

```bash
git add lambeq/text2diagram/model_based_reader/bobcat_parser.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py \
    benchmarks/RESULTS.md
git commit -m "Add opt-in multiprocess chart parsing to BobcatParser

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Final verification and combined benchmark

**Files:**
- Modify: `benchmarks/RESULTS.md`

- [ ] **Step 1: Run the full test suite for affected areas**

```bash
$PY -m pytest tests/test_bobcat.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all PASS.

- [ ] **Step 2: Lint the changed files (if flake8 is available)**

```bash
$PY -m flake8 lambeq/bobcat/tagger.py \
    lambeq/text2diagram/model_based_reader/bobcat_parser.py \
    benchmarks/bobcat_throughput.py || true
```

Fix any reported issues in the changed lines (repo style: 79-char lines).

- [ ] **Step 3: Combined-configuration benchmark**

Append under `## Combined (all optimizations)` in `benchmarks/RESULTS.md`:

```bash
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt \
    --max-spans-per-batch 5000 --dtype bfloat16 --n-jobs -1
# if CUDA available:
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt \
    --device cuda --max-spans-per-batch 20000 --dtype float16 --n-jobs -1
```

Use all 750 sentences (no `--num`). Add a short summary table at the top of
`benchmarks/RESULTS.md`: baseline vs combined, sent/s and peak memory, CPU
and GPU.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/RESULTS.md
git commit -m "Record combined benchmark results for Bobcat throughput work

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Report**

Summarize for the user: speedup factors per stage (CPU and GPU), memory
deltas, and which Tier-2 direction the numbers point to (chart parser
rewrite vs tagger work), per the spec's "Tier 2 outlook".
