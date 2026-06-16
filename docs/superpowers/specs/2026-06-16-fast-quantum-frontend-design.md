# Fast front-end for the quantum pipeline — design

**Date:** 2026-06-16
**Status:** approved
**Context:** The fast diagram core accelerates the *classical* tensor
pipeline (`fast-pipeline` branch: construct + remove_cups + FSpiderAnsatz
+ contraction). The quantum pipeline (`CircuitAnsatz` → quantum `Circuit`
→ pennylane/tket execution) shares the same FRONT-END: parse → diagram →
remove_cups, before the ansatz. `RemoveCupsRewriter` is especially
valuable for quantum (fewer cups = less post-selection = faster QML).
That front-end is exactly what the fast core already accelerates
(construction ~3×, fast `remove_cups` ~8.5×), so the quantum pipeline can
reuse it with no change to the ansatz or backends.

## Goal & non-goals

- **Goal:** a helper that produces the cup-reduced `grammar.Diagram` any
  `CircuitAnsatz` consumes, built via the fast front-end
  (`to_grammar(remove_cups(tree.to_fast_diagram()))`), byte-identical to
  the legacy front-end (`RemoveCupsRewriter()(tree.to_diagram())`), so
  quantum gets the construction + cup-removal speedups for free.
- **Non-goals:** a fast quantum ansatz (the `CircuitAnsatz` output is a
  quantum `Circuit`, not a tensor `FDiagram`; a fast version would need a
  circuit IR — deferred); quantum execution (pennylane/tket — different
  machinery, not the fast core's domain); any change to `CircuitAnsatz`,
  the quantum models, or `RemoveCupsRewriter`.

## Why this works (no fast quantum ansatz needed)

`CircuitAnsatz.__call__(diagram: grammar.Diagram) -> Circuit`
(circuit.py:114). It consumes a `grammar.Diagram`. The fast front-end
`to_grammar(remove_cups(tree.to_fast_diagram()))` produces the SAME
grammar diagram as `RemoveCupsRewriter()(tree.to_diagram())` — guaranteed
by the two existing byte-identical gates on `fast-pipeline`:
- construction: `to_grammar(t.to_fast_diagram()) == t.to_diagram()`,
- cup-removal: `to_grammar(remove_cups(to_fast(d))) == RemoveCupsRewriter()(d)`.
Their composition is the front-end identity this work asserts. The
quantum ansatz and backends then run unchanged.

## Component

`lambeq/backend/fast/pipeline.py` (next to `compile_fast_circuits`):

```python
def compile_quantum_input(trees):
    """Cup-reduced grammar.Diagrams for any CircuitAnsatz, built via the
    fast front-end (fast construction + fast remove_cups). Equivalent to
    `[RemoveCupsRewriter()(t.to_diagram()) for t in trees]`, faster."""
    return [convert.to_grammar(remove_cups(t.to_fast_diagram()))
            for t in trees]
```

Re-export from `lambeq/backend/fast/__init__.py`. Usage:
`circuits = [IQPAnsatz(ob, ...)(d) for d in compile_quantum_input(trees)]`.

## Testing & gates

1. **Front-end identity:** over the corpus,
   `compile_quantum_input(trees)[i] == RemoveCupsRewriter()(trees[i].to_diagram())`.
2. **Ansatz-circuit equivalence:** a real `IQPAnsatz` (a small
   `ob_map`, e.g. `{AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}`,
   n_layers=1) produces the SAME `Circuit` from the fast front-end as
   from the legacy front-end, over the corpus —
   `IQPAnsatz(...)(fast_in) == IQPAnsatz(...)(legacy_in)`. Proves the
   quantum circuit is byte-identical, so any downstream model/execution
   is unchanged. (Confirm `Circuit.__eq__` semantics; if circuits don't
   define structural `==`, compare via their grammar/box reprs.)
3. **Existing suites stay green** (additive; nothing else touched).

## Performance

`benchmarks/` (extend the pipeline bench or a tiny new one): front-end
ONLY — legacy `[RemoveCupsRewriter()(t.to_diagram())]` vs fast
`compile_quantum_input(trees)` — over the corpus, reporting the
construct + remove_cups speedup that now carries to quantum
preprocessing. The ansatz and execution are unchanged, so not measured.
Append a short note to `benchmarks/RESULTS.md`. Honest caveat: this
accelerates quantum *preprocessing* (diagram building), not circuit
execution (which dominates QML wall-clock and lives in the backend).

## Delivery (this spec → one short plan)

Branch `fast-quantum-frontend` (from `fast-pipeline`). One helper + the
two gates + the front-end benchmark. Files: `lambeq/backend/fast/
pipeline.py` (+`compile_quantum_input`), `__init__.py` re-export,
`tests/backend/test_fast_quantum_frontend.py`, benchmark + RESULTS note.
