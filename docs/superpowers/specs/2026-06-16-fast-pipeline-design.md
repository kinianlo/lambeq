# Unified fast pipeline (Tier 1) — design

**Date:** 2026-06-16
**Status:** approved
**Context:** The fast diagram core's stages were built and gated
individually but stranded on separate branches: direct construction
(`fast-construction`), `FSpiderAnsatz` (`fast-ansatz`), Rust construction
(`rust-diagram-core`), and the contraction/`FastPytorchModel` (on main).
No single branch ran the full fast pipeline, so the compounded throughput
win was never realised end to end. A pipeline profile (396 COCO diagrams,
laptop CPU) localised the legacy post-parser cost: ansatz 12.3 ms (63%),
`RemoveCupsRewriter` 4.7 ms (24%), construction 2.5 ms (13%) per diagram
preprocessing, plus 6.2 ms per training step. The fast core nearly
eliminates ansatz and construction; this work unifies them and closes the
remaining gap — `RemoveCupsRewriter` — with a fast-core port.

Branch `fast-pipeline` already merges `fast-construction` + `fast-ansatz`
(Python-only); Rust construction stays opt-in on `rust-diagram-core`
(folding it in would force a maturin build dependency for a marginal
construction gain).

## Goal & non-goals

- **Goal:** one branch running the complete fast classical pipeline —
  parse → `to_fast_diagram` → fast `RemoveCups` → `FSpiderAnsatz` →
  `FastPytorchModel` — numerically identical to the legacy pipeline, with
  the compounded throughput measured. Close the last slow stage
  (`RemoveCupsRewriter`) with an `FDiagram` port.
- **Non-goals:** Rust construction (opt-in elsewhere); cotengra / batched
  contraction (Tier 2); quantum circuits (the quantum pipeline keeps the
  legacy `RemoveCupsRewriter` — `remove_cups` is NOT deleted, and matters
  there because cups are expensive as circuit ops); merging to main
  (decided at finish); changing the legacy rewriter/ansatz/model (they
  stay as oracles).

## Decisions (made with user)

1. **Python-only unification** — merge construction + ansatz; Rust
   construction opt-in.
2. **Keep cup-removal, make it fast** — the user noted `remove_cups` is
   needed for quantum (expensive cups). The classical fast pipeline keeps
   a cup-removal stage but via a new fast-core `remove_cups` on
   `FDiagram` (the legacy `RemoveCupsRewriter` is itself a 4.7 ms
   object-churn stage). NOT dropped (even though the fast contraction
   handles cups, verified 10/10 — dropping it would diverge from the
   quantum pipeline and the legacy semantics the user wants preserved).
3. **Native fast model path** — `FastPytorchModel` consumes fast
   `FDiagram` circuits directly (no per-call `convert.to_fast`).
4. **End-to-end numeric gate** vs the full legacy pipeline.

## Components

### 1. Fast `remove_cups` (`lambeq/backend/fast/normal.py`)

Port `RemoveCupsRewriter` (rewrite/rewrite_diagram.py:108-192) onto the
`FDiagram` term arrays. The legacy is
`_remove_cups(_compress_cups(_remove_cups(d)))`:
- `_compress_cups`: merge adjacent nested `Cup`s into one wide cup
  (`CUP_TOKEN` box) — a single pass over `(box, offset)` with the
  nested-cup test `offset == prev_offset - 1`.
- `_remove_cups`: greedy contraction — for each box, walk the partial
  `diags` list to find the boxes to its left/right, and contract a cup by
  daggering+bending (`left.dagger().r`/`.l`, `right.dagger().l`/`.r`) or
  combine; non-cup boxes pass through.
These use `dagger`, `.l`/`.r`, `then_at`/`@`/`>>`, `Diagram.cups` — all
present on the fast core (`FBox.dagger`, `FTy.l/.r`, `FDiagram` ops,
`cups`). Add `remove_cups(d: FDiagram) -> FDiagram` mirroring the three-
pass structure. **Gate:** `to_grammar(remove_cups(to_fast(d))) ==
RemoveCupsRewriter()(d)` over the full corpus (differential, like the
snake-removal port). Re-export from `__init__.py`.

### 2. Native fast model path (`lambeq/backend/fast/model.py`)

`FastPytorchModel`:
- `_spec_for(diagram)`: if `diagram` is already an `FDiagram`, call
  `to_contraction(diagram)` directly (skip `convert.to_fast`); else keep
  today's legacy-tensor-circuit behaviour. Accepts both.
- `from_fast_diagrams(circuits)`: collect `Symbol`s from the `FBox`
  payloads of the fast circuits (sorted by name, mirroring how
  `PytorchModel.from_diagrams` collects `free_symbols`), set
  `self.symbols`. Provides the native entry; `initialise_weights` /
  checkpoint / training API inherited unchanged.

### 3. Pipeline helper (`lambeq/backend/fast/pipeline.py`)

One thin convenience function:
`compile_fast_circuits(trees, ob_map, max_order=2) -> list[FDiagram]` =
`[FSpiderAnsatz(ob_map, max_order)(remove_cups(t.to_fast_diagram())) for t in trees]`.
Minimal — no over-building. The full pipeline is then
`FastPytorchModel.from_fast_diagrams(compile_fast_circuits(trees, ob))`.

## Data flow

```
trees → t.to_fast_diagram()        (direct construction)
      → fast remove_cups           (FDiagram cup removal, new)
      → FSpiderAnsatz(ob)(...)      (fast ansatz, Symbol payloads)
      → FastPytorchModel.from_fast_diagrams(circuits)
      → train: get_diagram_output → cached to_contraction + evaluate
```
Legacy mirror (the oracle): `to_diagram → RemoveCupsRewriter →
SpiderAnsatz → PytorchModel.from_diagrams → get_diagram_output`.

## Testing & gates

1. **Fast `remove_cups` differential:** `to_grammar(remove_cups(
   to_fast(d))) == RemoveCupsRewriter()(d)` over the corpus. 0 mismatches.
2. **End-to-end numeric gate (the headline):** over the corpus (filtered
   to a uniform output shape so `PytorchModel` can stack), the full fast
   pipeline's `get_diagram_output` equals the full legacy pipeline's,
   `allclose` atol 1e-5, weights matched by `Symbol` name (seeded). This
   proves construction + remove_cups + ansatz + contraction compose
   correctly.
3. **Native-model unit checks:** `from_fast_diagrams` collects the same
   symbol set as `PytorchModel.from_diagrams` on the equivalent legacy
   circuits; `_spec_for` accepts an `FDiagram` without `convert.to_fast`.
4. **Existing per-component suites stay green** (build, construct,
   ansatz, contraction, rule-layer).

## Performance

`benchmarks/fastpipeline_bench.py`: full legacy pipeline vs full fast
pipeline over the corpus — preprocess ms/diagram (construct + remove_cups
+ ansatz) and per-step ms (get_diagram_output), legacy vs fast, the
compounded speedup. Appended to `benchmarks/RESULTS.md`. Honest verdict;
note the laptop-CPU parser caveat (this win lands in the GPU-parsing /
large-scale regime). Pre-assert numeric equivalence before timing.

## Delivery (this spec → one plan)

Branch `fast-pipeline` (already merges construction + ansatz). TDD:
fast `remove_cups` + its differential gate first; then the model path +
unit checks; then the end-to-end numeric gate; then the helper +
benchmark. Files: `lambeq/backend/fast/normal.py` (+remove_cups),
`model.py` (native path), new `pipeline.py`, `__init__.py` re-exports,
`tests/backend/test_fast_pipeline.py`, `benchmarks/fastpipeline_bench.py`,
`RESULTS.md`.
