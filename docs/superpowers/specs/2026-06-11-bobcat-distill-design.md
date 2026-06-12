# Tier 4: distilled Bobcat supertagger — design

**Date:** 2026-06-11
**Status:** approved (user granted full design authority)
**Predecessors:** Tiers 1-3 (`2026-06-10-bobcat-throughput`,
`2026-06-10-bobcat-rust-cky`, `2026-06-11-bobcat-tagger-speed`)

## Background

After Tier 3, the 30-layer hidden-1024 BERT encoder is ~75% of
end-to-end GPU time (forward 54ms of a ~109ms per-64-batch cycle).
The last large lever is a smaller encoder. The original model was
trained on CCGBank (LDC-licensed, not available here); however,
distillation needs no gold labels — the teacher labels unlabeled text.

## Goals & non-goals

- **Goals:** a student supertagger with >=3x faster encoder forward,
  >=95% tree-level agreement with the teacher on held-out text, packaged
  as a drop-in Bobcat model directory (works with
  `BobcatParser(model_name_or_path=<dir>)` and all Tier 1-3 machinery).
- **Non-goals:** matching CCGBank gold accuracy (unmeasurable here —
  agreement WITH THE TEACHER is the quality bar); quantization; encoder
  architecture changes beyond depth; upstreaming.

## Decisions

1. **Data: unlabeled wikitext-103** (free download). Sentences of 4-60
   words after cleaning; ~2-4M available, target 500k-1.5M for
   training, 2000 held out for the agreement gate (never trained on),
   plus the existing 937-sentence benchmark corpora as a secondary,
   distribution-shifted check. Data prepared on the laptop (reliable
   internet), shipped to the project share as JSONL (one tokenised
   sentence per line).
2. **Student: depth-pruned teacher.** `BertForChartClassification` with
   `num_hidden_layers=6`, hidden 1024, same tokenizer/vocab. Initialise
   embeddings, tag/span heads, and final layer norm from the teacher;
   encoder layers from equally spaced teacher layers {0, 6, 12, 17,
   23, 29}. Depth is a config knob (`--layers`); 12 is the fallback if
   6 misses the gate.
3. **Loss: temperature-scaled KL** (tau = 2.0):
   `loss = (n_tags * KL_tags + n_spans * KL_spans) / (n_tags + n_spans)`
   over REAL positions only (padding masked), each KL averaged within
   its group and scaled by tau^2, mirroring the teacher's own
   count-weighted loss structure. Teacher runs under
   `inference_mode` + fp16 autocast; student trains in bf16 autocast
   with fp32 master weights (standard AMP).
4. **Training:** online distillation (teacher and student in the same
   loop — no precomputed logits; span logits are too large to store).
   AdamW lr 1e-4, cosine decay to 0, 1000-step warmup, grad clip 1.0,
   length-sorted token-budget batching (target ~4096 words/batch),
   checkpoint (model + optimizer + scheduler + data cursor) every 1000
   steps, fully resumable. Periodic cheap dev metric: tag top-1
   agreement on a fixed 64-sentence dev batch.
5. **Compute: beaker A40 (gpu.q, tmem 14G, h_rt 23:59)** with
   checkpoint-resume across jobs if needed; everything (data,
   checkpoints, logs) on `/SAN/intelsys/discoviz/kinianlo/lambeq-bench/
   distill/`. Estimated step cost (~64 sentences avg 25 words): teacher
   fwd ~30ms + student fwd/bwd ~30ms + overhead → ~5-10k sentences/min;
   3 epochs over 500k sentences fits in a few hours.
6. **Acceptance gate:** on the 2000 held-out sentences, parse with
   teacher-Bobcat and student-Bobcat (both `parser_backend='rust'`,
   identical settings): tree agreement (CCGTree equality, counting
   None==None as agreement only when both fail) **>= 95%**. Secondary
   report (no gate): agreement on the 937 benchmark sentences, tag
   top-1 agreement, and parse-failure-rate delta. If 6 layers < 95%,
   retrain at 12 layers before concluding.
7. **Packaging:** `package_student.py` writes a complete model dir:
   student weights + config.json (incl. tags/cats lists), tokenizer
   files, `grammar.json`, `pipeline_config.json`, `version.txt` copied
   from the teacher dir. Benchmark via
   `bobcat_throughput.py <corpus> --device cuda --parser-backend rust`
   with `BobcatParser(model_name_or_path=<student>)` (benchmark gains a
   `--model-dir` flag).

## Components (all under `tools/distill/`)

- `prepare_data.py` — download wikitext-103-raw (HF `datasets` if
  available, direct URL fallback), clean (strip headings/empty lines,
  unescape), sentence-split (regex on sentence punctuation — spaCy not
  required), whitespace-tokenise, filter 4-60 words, dedup, shuffle
  (seed 0), write `train.jsonl`, `heldout.jsonl` (2000), `dev.jsonl`
  (64).
- `make_student.py` — build + save the initialised student from the
  teacher dir (`--layers 6`, `--layer-map` computed as equally spaced).
- `distill.py` — the training loop (args: data dir, teacher dir,
  student dir, out dir, batch token budget, lr, steps/epochs, tau,
  checkpoint-every, resume). Single GPU. Logs JSONL metrics.
- `eval_agreement.py` — tree-agreement gate (teacher dir vs student
  dir, sentence file, reports agreement %, both-fail %, tag top-1
  agreement; exit code reflects the 95% gate when `--gate` passed).
- `package_student.py` — final model dir assembly.
- Benchmark: `--model-dir` flag added to `benchmarks/bobcat_throughput.py`.

## Error handling

- `distill.py` resumes exactly from checkpoint (model/optimizer/
  scheduler/data cursor/step count); SIGTERM (SGE h_rt kill) loses at
  most the last interval.
- All scripts validate paths and fail loudly; no silent fallbacks.
- The SGE job script resubmits itself when training exits with the
  "not finished" sentinel code (max 5 resubmissions).

## Testing

- Local CPU smoke: `make_student.py --layers 2` + `distill.py` for 5
  steps on 50 sentences + `eval_agreement.py` on 10 sentences — wired
  as a pytest (`tests/test_distill_smoke.py`, marked slow, skipped in
  normal runs via an env guard) and run manually before launching.
- The packaged student dir must pass: `BobcatParser(
  model_name_or_path=<dir>, parser_backend='rust')` parses the 5
  standard test sentences without error (part of eval_agreement).

## Risks

- Span-head KL over O(n^2) positions is the memory-heavy part of the
  loss: with token-budget batching at 4096 words and 60-word cap, span
  positions per batch stay ~bounded (~2k rows x 968); bf16 keeps it
  well under the A40's 48GB. If OOM: halve the token budget (knob).
- Wikitext domain shift vs the teacher's training data is irrelevant
  for the agreement gate (both models see the same input) but means
  gold-accuracy claims are out of scope (already a non-goal).
- 95% tree agreement at 6 layers may simply be unreachable — the
  12-layer fallback is the designed response; if that also fails, stop
  and report honestly.

## Deliverables

1. `tools/distill/` scripts + smoke test, committed.
2. A trained, packaged student model dir on the project share (and the
   training log).
3. RESULTS.md section: agreement numbers, throughput before/after
   (teacher vs student on goosander), verdict against the >=3x encoder
   target.
