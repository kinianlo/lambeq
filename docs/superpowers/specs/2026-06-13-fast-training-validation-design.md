# Fast diagram core — training validation & benchmark — design

**Date:** 2026-06-13
**Status:** approved
**Context:** The fast diagram core (`lambeq/backend/fast/`) merged to main
(62e3d6f). `FastPytorchModel` is a drop-in `PytorchModel` subclass whose
forward pass is numerically gated against the stock model, and a one-off
gradient check confirmed gradients flow and match (maxdiff 1.7e-6 over
1328 params). What is NOT yet demonstrated: that the fast path trains
end-to-end — the same loss trajectory under a real optimizer loop — and
what the actual per-epoch wall-clock speedup is. This work supplies that
proof, self-contained inside the lambeq repo.

## Goal & non-goals

- **Goal:** Prove `FastPytorchModel` trains identically to `PytorchModel`
  (same seed/data/optimizer → matching loss trajectory) and measure the
  per-epoch wall-clock speedup, on a real parse→ansatz→train pipeline.
- **Non-goals:** classifier accuracy (the label is synthetic; we validate
  trajectory-match and speed, not task performance); touching discoviz
  (it has its own cotengra `EinsumModel`); using lambeq's `PytorchTrainer`
  (a minimal hand-rolled loop is used for full control of seed/data/
  timing); contraction-path optimization (the greedy pairwise engine is
  measured as-is; cotengra adoption is a separate future task).

## Decisions (made with user)

1. **Self-contained lambeq demo** — lives in the repo we own; does not
   perturb discoviz's working production pipeline.
2. **COCO captions + synthetic labels** — real text through the real
   cached-Bobcat → remove_cups → SpiderAnsatz pipeline; labels are a
   deterministic function of the caption (reproducible, no new data
   dependency). Corpus: `/tmp/coco_bench.txt` (2000 captions already on
   disk).
3. **Minimal hand-rolled training loop** — explicit forward/loss/backward/
   step; same optimizer (Adam), seed, and data order for both models.
4. **Add a CI gate** — a small pytest asserting the trajectory match, so
   "training stays correct" is locked, not just script-measured.

## Pipeline & data flow

1. Read N captions from `/tmp/coco_bench.txt`. Assign a **deterministic
   binary label** by keyword presence (a small fixed keyword set chosen
   for rough balance, e.g. people/animal words → 1 else 0). The label
   function is pure and seedless.
2. Parse with `BobcatParser(parser_backend='rust')`, drop unparseable
   sentences, then `RemoveCupsRewriter()` → `SpiderAnsatz` mapping the
   sentence type to `Dim(2)` (other atomic types to a fixed `Dim`, e.g.
   `Dim(2)`).
3. **Filter to a stackable output shape.** `PytorchModel.get_diagram_output`
   does `torch.stack` over per-diagram outputs, which requires a uniform
   cod dimension. Keep only circuits whose contracted output shape equals
   the modal/target shape (the sentence-type shape); report how many of N
   were kept. This filtering is shared by both models so they see the
   identical circuit set.
4. Build `PytorchModel.from_diagrams(circuits)` and
   `FastPytorchModel.from_diagrams(circuits)`. Initialise the stock model
   under a fixed seed, then **clone its weight values** (`.detach().clone()`
   per parameter) into the fast model so both start at an identical point
   and train as independent parameter sets (not shared tensors).
5. **Train both** for E epochs with separate `torch.optim.Adam` instances
   (same lr), identical batch/data order, loss =
   `mse(stack(outputs), labels_onehot)` (or BCE on a 2-logit output —
   pick one and fix it; MSE against one-hot is the default). Record
   per-epoch mean loss and per-epoch wall-clock for each model.

## Components

- `benchmarks/fasttrain_bench.py` — the script. CLI like the existing
  `benchmarks/bobcat_throughput.py` / `fastdiag_bench.py`: positional
  corpus path, `--num` (cap sentences), `--epochs`, `--lr`, `--seed`.
  Prints (a) circuits kept / N, (b) a per-epoch loss table old vs fast
  with the max abs trajectory difference, (c) a wall-clock table
  (total + per-epoch, old vs fast, speedup). Exits nonzero if the
  trajectories diverge beyond tolerance (so the script doubles as a gate).
- `tests/backend/test_fast_training.py` — a fast CI gate: a handful of
  sentences (e.g. 12 parseable), 2–3 epochs, asserts the two loss
  trajectories are `allclose` (atol 1e-4). Runs with validation ON via the
  existing `tests/backend/conftest.py` fixture.
- `benchmarks/RESULTS.md` — append a `## Fast diagram core — training`
  section with the kept-circuit count, the trajectory-match result, and
  the wall-clock table, with honest verdicts (per campaign convention).

## Why the trajectories must match (and the gate's logic)

Gradients already match to 1.7e-6 for a single backward. With identical
initial weights, identical optimizer state, and identical data order, each
optimizer step applies the same update, so the loss sequences coincide for
all epochs up to float noise. A divergence therefore signals a real bug
(nondeterministic ordering, weight aliasing, or a gradient discrepancy that
only compounds under iteration) — which is exactly what the gate catches.

## Testing & gates

1. **Trajectory-match gate (CI):** `test_fast_training.py` —
   `allclose(loss_old, loss_fast, atol=1e-4)` across epochs on a small set.
2. **Benchmark self-check:** the script asserts the same and exits nonzero
   on divergence, so the recorded numbers are never from a broken run.
3. **Existing suites stay green** (no changes to library code expected;
   this is additive — script + test + RESULTS only).

## Performance reporting

Report per-epoch wall-clock old vs fast and the speedup, on whatever N the
run used (note the count). Expectation set honestly: the fast per-step win
was 1.4× on the full-corpus contraction benchmark and ~3× on short
sentences; the training loop adds optimizer/loss overhead common to both,
so the end-to-end epoch speedup will be bounded by the contraction share of
each step. Misses or surprises get a one-line profile, per campaign
convention. One-off costs (spec extraction, ansatz) are reported separately
from the per-epoch steady state.

## Delivery

Single plan: script + test + RESULTS, on branch `fast-training-validation`
(forked from main 62e3d6f). Additive only.
