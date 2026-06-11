# Tier 3: faster Bobcat supertagger — design

**Date:** 2026-06-11
**Status:** approved
**Predecessors:** `2026-06-10-bobcat-throughput-design.md` (Tier 1),
`2026-06-10-bobcat-rust-cky-design.md` (Tier 2, `bobcat_rs`)

## Background and evidence

After Tier 2, the supertagger dominates wall time everywhere (90-95% of
end-to-end on GPU). Component profile on goosander (RTX 3090 Ti), one
64-sentence batch of 25-50-word sentences, fp16:

- BERT body forward: 46.5 ms
- classification heads: +6.6 ms
- `extract_topk` post-processing: **54.7 ms** — as expensive as the
  whole GPU forward, despite the Tier-1 survivor-only transfer.
- Attention already uses SDPA.

Config sweep (long corpus, 187 sentences): batch 16/fp32 (current
default path) 304 sent/s; batch 64/fp16 **623 sent/s**. Batch 128/fp16
is slower (535) — ~64 is the long-sentence sweet spot.

Decisions (with user): do (a) tuned GPU defaults + (b) raw-tensor
handoff to Rust + (c) torch.compile AND an ONNX Runtime lane. GPU
defaults switch automatically (this is a fork; explicit settings always
win). Distillation and TensorRT are out of scope.

## Goals & non-goals

- **Goals:** long-sentence GPU tagging from ~300 to >=1000 sent/s;
  fused tagger->parser lane with no Python-object churn; opt-in
  compile/ONNX acceleration; outputs identical on the torch lane
  (boundary-tie tolerance only on the opt-in ONNX lane).
- **Non-goals:** distillation/smaller models; TensorRT; forward/parse
  pipelining (revisit only if post-handoff profiling shows
  post-processing still >15% of tagger time); CPU-path changes beyond
  what falls out naturally.

## Work items

### 1. Tuned GPU defaults

In `BobcatParser._initialise_model`: when the tagger device type is
`cuda` and the user did not explicitly pass them (and
`pipeline_config.json` is at its shipped values), default
`dtype='float16'` and `batch_size=64`. Explicit kwargs and CPU
behaviour unchanged. Documented in the `BobcatParser` docstring.
Implementation note: "explicitly passed" is detected in
`_initialise_model` before the pipeline-config merge — the kwargs dict
is consulted for the keys prior to popping them into the config.

### 2. Raw-tensor handoff (core item)

New fused lane used by `sentences2trees` when the Rust parser backend
is active:

- `Tagger.forward_topk(batch_sentences) -> RawTagBatch`: runs
  `prepare_inputs` + model forward + `log_softmax().topk(k)` on-device
  (k = the configured tag/span top-k, as in `extract_topk`), then moves
  to CPU exactly five contiguous tensors per batch: tag scores
  (f32 [B, W, k_tag]), tag indices (i64), span scores
  (f32 [B, S, k_span]), span indices (i64), plus the per-sentence word
  counts. No threshold, no Python tuples. Word strings travel
  separately (list of lists).
- `bobcat_rs.RustChartParser.parse_batch_raw(words, tag_scores,
  tag_indices, span_scores, span_indices, lengths, num_threads)`:
  accepts the buffers zero-copy via the PyO3 `numpy` crate; applies
  thresholding with EXACTLY `extract_topk` semantics (relative/absolute
  strategies, prob thresholds, span index-0 exclusion, padding-position
  masking, descending-order prefix property) and the supertag/span-score
  assembly semantics of `_prepare_sentence` (tag index -> category
  string via the model `tags` list, supplied at construction; span
  buckets keyed by `idx2span` ordering replicated in Rust); then parses
  as today. Threshold config and the `tags` list become constructor
  inputs of `RustChartParser` (new parameters; the Python-side
  `RustBackend` passes them from the tagger config).
- The classic lane (`Tagger.__call__` -> `TaggerOutput` ->
  `_prepare_sentence`) remains intact for `parser_backend='python'`,
  external `Tagger` users, and the C&C text export.

Equivalence gates:
- Differential test: Rust thresholding+assembly == `extract_topk` +
  `_prepare_sentence` over random logits, parametrised over strategies,
  thresholds, top-k values, and mixed sentence lengths.
- The 937-sentence corpus checker (`benchmarks/check_equivalence.py`)
  extended to compare fused-lane trees vs the Python backend: must stay
  937/937 on the torch lane.

### 3. torch.compile (opt-in)

`BobcatParser(compile_model=True)` wraps the BERT body
(`tagger.model.bert`) with `torch.compile(dynamic=True)` (sorted
batching produces varying [B, seq] shapes; dynamic avoids recompile
churn). Default False — first-call compilation latency (~minutes) only
pays off for corpus-scale runs. Plumbed like other post-config kwargs;
benchmark flag `--compile-model`.

### 4. ONNX Runtime lane (opt-in)

- `tools/export_onnx.py`: exports `BertForChartClassification` (two
  outputs: tag_logits, span_logits) with dynamic batch/sequence axes to
  `<model_dir>/bobcat.onnx`, taking the model dir as argument. The
  span-classifier input construction (word-mask gather + span pairing)
  must be inside the exported graph; if the gather proves unexportable,
  export the BERT body only and keep heads in torch (decide in
  implementation; either satisfies the lane's contract: logits out).
- `BobcatParser(tagger_backend='onnx')`: the tagger forward runs via
  `onnxruntime-gpu` (CUDA execution provider, fp16 where supported),
  producing numpy logits that feed the same topk + handoff path
  (topk computed in torch on numpy-wrapped tensors or via ORT —
  implementation's choice, gated by the differential tests).
  Default `tagger_backend='torch'`. Missing onnxruntime -> ImportError
  with install hint. `'onnx'` + missing export file -> FileNotFoundError
  naming the export script.
- Acceptance: corpus checker mismatches <=1% on the ONNX lane
  (boundary ties under different kernel arithmetic), 0 on torch lanes.

### 5. Benchmarks

`benchmarks/bobcat_throughput.py` gains `--compile-model` and
`--tagger-backend`. Matrix on laptop CPU, goosander (3090 Ti), beaker
(A40): defaults-only, +handoff, +compile, +onnx. Results appended to
`benchmarks/RESULTS.md` with a post-handoff component re-profile
(forward vs post-processing share) to settle the pipelining question.

## Error handling

- Invalid `tagger_backend` / `compile_model` values: ValueError at
  construction.
- `parse_batch_raw` shape mismatches (B/W/S inconsistent with words):
  ValueError from Rust via PyO3, not a panic.
- The fused lane preserves `suppress_exceptions` semantics including the
  whole-batch BaseException guard added for Rust panics.

## Testing

- New differential tests in `tests/test_bobcat_rs.py` (thresholding/
  assembly equivalence; fused-lane tree equality on the standard
  5-sentence set).
- Existing 64-test suite green under `parser_backend` python and rust,
  with and without the new GPU defaults active (CPU CI machines:
  defaults don't trigger).
- Corpus gates per work item 2 and 4.

## Performance targets (long corpus, 3090 Ti)

- Defaults alone: >=600 sent/s tagging (measured 623).
- + raw handoff: >=900 sent/s end-to-end tagging stage.
- + compile or ONNX: stretch >=1200 sent/s; record honestly whichever
  wins; if compile/ONNX individually deliver <10%, say so and keep them
  opt-in (they remain useful on other hardware).

## Delivery phases

1. GPU defaults + benchmark flags (small, immediate win).
2. Raw handoff: Tagger.forward_topk + parse_batch_raw + gates.
3. torch.compile flag.
4. ONNX export + lane + gates.
5. Benchmark matrix on three machines + RESULTS + re-profile.
