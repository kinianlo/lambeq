# BobcatParser throughput improvements — design

**Date:** 2026-06-10
**Status:** approved

## Background

`BobcatParser` is a two-stage pipeline:

1. **Supertagger** (`lambeq/bobcat/tagger.py`): a BERT encoder (30 layers,
   hidden size 1024, ~400M params) with two heads — a per-word tag
   classifier (1285 tags) and a span classifier scoring every O(n²) chart
   span (968 categories each, from 2048-dim concatenated endpoints).
2. **Chart parser** (`lambeq/bobcat/parser.py`): pure-Python CKY with a
   beam of 32, run sequentially per sentence on the main thread.

Profiled/inspected problems:

- The tagger forward pass runs **without `torch.no_grad()` /
  `inference_mode()`** (`tagger.py`, `Tagger.parse`) — every batch builds
  a full autograd graph. (`OncillaParser` already uses `no_grad`.)
- Default `batch_size=4` with no length grouping: batches pad to the
  longest member and span-head cost grows quadratically with padded
  length.
- Post-processing runs `log_softmax().topk()` then nested Python loops
  over `.tolist()` with `span_top_k=128` — slow per-token Python work.
- Once the tagger runs on GPU, the single-threaded pure-Python chart
  parser dominates wall time, despite being embarrassingly parallel
  across sentences.
- fp16/bf16 inference is unused.

## Goals & non-goals

- **Goals:** higher sentences/sec and lower peak memory, on both CPU and
  GPU. All Tier-1 changes upstream-mergeable: pure Python + torch, no new
  heavy dependencies, public API stable (new kwargs are opt-in with
  defaults preserving current behaviour).
- **Non-goals:** accuracy improvements; bit-identical outputs (mild
  numerical deviation is acceptable when the user opts into reduced
  precision); Tier-2 rewrites (see Outlook).

## Tier 1 work items

### 0. Benchmark harness (first)

`benchmarks/bobcat_throughput.py`, a standalone script (not part of the
package). Input: a sentence file + device (+ parser kwargs). Output:
tagging time, chart-parsing time, sentences/sec, peak CPU RSS, peak CUDA
memory — reported separately per stage. Every subsequent item is measured
before/after with this script on CPU and on a GPU (RTX 3080). It also
provides the evidence for choosing Tier-2 work.

### 1. `torch.inference_mode()` in `Tagger.parse`

Wrap the model forward call. No API change. Expected: the largest memory
reduction (no autograd graph over the 30-layer BERT and O(n²) span head)
and a meaningful speedup on both devices.

### 2. Length-sorted batching in `Tagger.__call__`

Sort sentence indices by token count, batch over the sorted order,
restore the original order before returning. Per-sentence output and
output order are unchanged. Removes quadratic padding waste in the span
head for mixed-length input.

### 3. Span-budget batching

New tagger param `max_spans_per_batch: int | None = None`.

- `None` (default): current fixed `batch_size` behaviour.
- Set: batches accumulate sentences until the padded span count
  (`batch_len × chart_size(max_len_in_batch)`) would exceed the budget —
  large batches for short sentences, small for long ones, flat memory
  profile. Composes with item 2 (sorting makes batches
  length-homogeneous).

### 4. Vectorized tag/span post-processing

Replace the nested Python loops in `Tagger.parse` with tensor ops:
compute the threshold mask on-device (relative/absolute strategies, the
span index-0 exclusion), then transfer only surviving entries to CPU in
one `.tolist()` call and assemble the output lists. Output format
(`TaggerOutputSentence`) and values unchanged — verified by an
equivalence test against the current implementation.

### 5. Opt-in reduced precision

New kwarg on `BobcatParser`/`Tagger`: `dtype` (e.g. `'bfloat16'`,
`'float16'`; default `None` = current fp32). When set, the forward pass
runs under `torch.autocast` for the model device. Default behaviour
unchanged (upstream-safe); GPU users opt in for speed.

### 6. Opt-in parallel chart parsing: `n_jobs`

`sentences2trees(..., n_jobs: int = 1)`; `n_jobs=-1` = all cores.

- Implementation: `multiprocessing.Pool` over tagged sentences. On
  fork-capable platforms the workers inherit the constructed
  `ChartParser` cheaply; spawn platforms use a per-worker initializer.
- `suppress_exceptions` semantics preserved: a failed sentence yields
  `None` (suppressed) or re-raises as `BobcatParseError`.
- Risk to verify early: `CCGTree`/`ParseTree` pickling for returning
  results from workers (the `metadata['original']` field holds the
  Bobcat `ParseTree`).

## Error handling

- Invalid `max_spans_per_batch`, `n_jobs`, or `dtype` raise `ValueError`
  at construction, matching the existing validation style in
  `Tagger.__init__`.
- All other error semantics unchanged.

## Testing

- Existing `tests/test_bobcat.py` and
  `tests/text2diagram/model_based_reader/test_bobcat_parser.py` pass
  untouched.
- New tests:
  - sorted batching restores original sentence order;
  - vectorized post-processing output equals the old loop output on a
    fixed input batch;
  - `n_jobs=2` produces the same trees as `n_jobs=1`;
  - span-budget batching never exceeds the budget (except for a single
    sentence exceeding it alone, which forms its own batch).

## Tier 2 outlook (not designed here)

Fork-tier, chosen by benchmark evidence after Tier 1 lands, depending on
which stage remains the wall:

- chart parser: Cython/Rust CKY core;
- tagger: ONNX Runtime / int8 quantization / `torch.compile`;
- model: distilled smaller supertagger.
