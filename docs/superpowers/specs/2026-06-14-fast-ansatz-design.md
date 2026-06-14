# Fast SpiderAnsatz (`FFunctor`-based) — design

**Date:** 2026-06-14
**Status:** approved
**Context:** The fast diagram core (`lambeq/backend/fast/`) is on main:
immutable `FDiagram`, copy-free `FFunctor` (125x the legacy
`grammar.Functor` on the structural map), `to_contraction`/`evaluate`
(the einsum path), numerically gated vs `PytorchModel`. Profiling the
post-parser preprocessing (300 COCO diagrams, i7-11800H) found the
**SpiderAnsatz is the single biggest remaining cost at ~2.06 ms/diagram**
(vs construction 0.76 ms, `to_contraction` 0.05 ms). It is slow for the
same reason construction was: `SpiderAnsatz` runs on the legacy
`grammar.Functor`, paying the `grammar.Diagram` object-churn (mutability,
per-op validation, repr-hashing) the fast core eliminates. The fix is the
same playbook that worked for construction: do it on the fast core
(`FFunctor` + `FDiagram`) in pure Python — no Rust (a profiled Rust
assessment showed the cost is object churn, not compute, so Rust would be
marginal; that is the deferred fully-fused-lane discussion).

This is the pure-Python form of "stage B" from the all-Rust roadmap.
Stage C (`to_contraction`, 0.05 ms, cached per diagram in training) is NOT
worth optimizing and is out of scope.

## Goal & non-goals

- **Goal:** `FSpiderAnsatz`, a pure-Python `FFunctor`-based ansatz mapping
  `FDiagram -> FDiagram` (Dim-typed wires + `Symbol` payloads on
  parameterized boxes), producing a numerically-identical pipeline to the
  legacy `SpiderAnsatz` and capturing the object-churn share of the
  ~2.06 ms (target: ~0.1-0.2 ms/diagram).
- **Non-goals:** other ansätze (`TensorAnsatz` base, MPS, IQP/quantum
  circuits) — `SpiderAnsatz` only; stage C / `to_contraction`
  optimization (already fast + cached); Rust (object-churn win is
  Python-capturable; the fully-fused all-Rust lane is a separate future
  campaign); changing the legacy `SpiderAnsatz` (it stays as the oracle)
  or the fast core data model.

## Decisions (made with user)

1. **Pure Python, `FFunctor`-based** — the win is object churn, captured
   by running on the fast core; Rust deferred.
2. **Single `FFunctor` pass** — `FFunctor.ar` may return a whole
   `FDiagram`, so the split (high-arity → spider chains) and the
   dim+symbol assignment fold into one pass (the legacy ansatz uses two
   functors: split then dims).
3. **Numeric-equivalence gate** vs `PytorchModel` (Symbol payloads are
   distinct objects, so naive `FBox ==` won't compare across paths;
   numeric identity through the contraction path is the real gate).
4. **`SpiderAnsatz` only**, default `max_order=2`.

## Architecture & data flow

`FSpiderAnsatz(ob_map: Mapping[grammar.Ty, Dim], max_order=2)` builds one
`FFunctor(ob, ar)`:

- **`ob`** (atom id -> `FTy` of Dim atoms): look up the atom's base type
  in `ob_map`, map to its `Dim`, intern each Dim integer via
  `register_dim` (the `#<int>` side-table `convert.py` uses), return the
  Dim-atom `FTy`. Adjoint atoms (z != 0) map to the SAME Dim as the base
  (matches `Dim.rotate` = identity for atomic Dims). Implemented by
  recovering `(name, z)` from the atom (`atom_name`/`atom_z`), looking up
  `ob_map[grammar.Ty(name)]` (base, z-stripped).

- **`ar`** (`FBox -> FDiagram`): mirrors `SpiderAnsatz._split_ar` +
  `TensorAnsatz._ar`:
  - Structural boxes (CUP/CAP/SWAP/SPIDER): re-emit the same kind with
    dim-mapped dom/cod, `payload=None`. (Unlike legacy `tensor.Cap`, the
    CAP array is NOT stored — `to_contraction` synthesizes the cap delta
    from `kind=CAP`.)
  - WORD/PLAIN box: if `len(dom) + len(cod) <= max_order`, emit ONE box
    with a `Symbol` payload (name + directed dims below) and dim-mapped
    dom/cod. Else split:
    - non-empty dom (uncurry): reproduce `CollapseDomainRewriteRule`
      (left): `Id(dom) @ Box(name, (), dom.r @ cod) >> Cup(dom, dom.r) @
      Id(cod)`, then recurse the split on the new wide-cod box.
    - wide cod: chain — for each `step_size = max_order - 1` slice of the
      cod, a `Word(f'{name}_{i}', cod_slice)` symbol box, joined by
      `Spider(cod_slice[-1:], 2, 1)` boxes. Mirror `_split_ar` exactly
      (offsets/order); the numeric gate confirms it.
  - The split fragment is returned as an `FDiagram` over the mapped
    (Dim) types, with `Symbol` payloads on its word boxes and bare
    `SPIDER`/`CUP`/`CAP` boxes for the structural parts.

- **`Symbol` creation** (uses lambeq `Symbol`): name from a port of
  `BaseAnsatz._summarise_box` (`f'{box.name}_{dom}_{cod}'`, with the
  `:, ` -> `\:`,`\,`,`\ ` escaping); `directed_dom`/`directed_cod` from a
  port of `TensorAnsatz._generate_directed_dom_cod` (per-leg flow by
  z-parity). These must be byte-identical to the legacy ansatz so
  `Symbol.size` and `PytorchModel` weight allocation match.

`FSpiderAnsatz.__call__(fd)` = `self.functor(fd)`. Output feeds directly
into `to_contraction` (the existing einsum path).

## Components / files

- `lambeq/backend/fast/ansatz.py` (new): `FSpiderAnsatz`, the
  `ar`/`ob` closures, and the ported `_summarise_box`/`_directed_dom_cod`
  helpers. One focused module.
- `lambeq/backend/fast/__init__.py`: re-export `FSpiderAnsatz`.
- Test `tests/backend/test_fast_ansatz.py`.

## Testing & gates

1. **Numeric-equivalence gate (the real one):** over the corpus, build
   circuits both ways and assert identical pipeline tensors —
   - legacy: `SpiderAnsatz(ob)(remove_cups(d))` -> `PytorchModel.
     from_diagrams` -> `initialise_weights` (seeded) -> `get_diagram_output`.
   - fast: `FSpiderAnsatz(ob)(to_fast(remove_cups(d)))` -> `to_contraction`
     -> `evaluate`, weights matched to the legacy model's by `Symbol`
     name.
   - `torch.allclose(fast, legacy, atol=1e-5)` over the corpus. This
     reuses the existing `test_fast_contraction` harness and proves the
     ansatz (including splitting) is numerically identical end to end.
2. **Symbol-parity unit check:** for a set of boxes (incl. a wide-cod
   word that triggers splitting), the Symbol names and `.size`s produced
   by `FSpiderAnsatz` match those from the legacy `SpiderAnsatz` (set
   equality over `{(sym.name, sym.size)}`).
3. **Structural unit tests:** ob maps atoms/adjoints to the right Dim;
   a small known box maps to the expected boxes (kinds, dims); a wide-cod
   box splits into the expected spider chain shape.
4. **Existing suites stay green** — legacy `SpiderAnsatz` and the fast
   core untouched; additive only.

## Performance

Benchmark over the corpus: legacy `SpiderAnsatz(d)` vs
`FSpiderAnsatz(to_fast(d))` ms/diagram + speedup, appended to
`benchmarks/RESULTS.md`. Honest verdict vs the ~0.1-0.2 ms target and a
one-line profile of any residual (e.g. Symbol object construction is the
irreducible Python cost). Note that this is a preprocessing
(once-per-dataset) win, not a training-step win.

## Delivery (this spec -> one plan)

Branch `fast-ansatz` (from main). TDD: structural + symbol-parity unit
tests first, then the corpus numeric gate, then the benchmark. Files:
new `lambeq/backend/fast/ansatz.py`, `__init__.py` re-export,
`tests/backend/test_fast_ansatz.py`, `benchmarks/` (bench + RESULTS).
