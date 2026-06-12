# Fast diagram core (`lambeq.backend.fast`) — design

**Date:** 2026-06-12
**Status:** approved
**Context:** upstream lambeq is unmaintained; this fork is the final
destination. No upstream-compatibility constraint.

## Background and evidence

The diagram representation (`lambeq/backend/grammar.py`) is the
dominant cost of every post-parser stage. Measured on 300 COCO-caption
diagrams (laptop CPU):

| operation | today | root cause |
|---|---:|---|
| tree -> diagram        | 2.7 ms | 20k Python calls/diagram; `Ty.tensor` x110k, `Ty.__iter__` x1.2M |
| remove_cups            | 3.9 ms | Functor machinery + repr-hash cache keys + pickle on cache hits |
| ansatz functor         | 7.5 ms | same + double dom/cod validation per box |
| fast_deepcopy          | 1.4 ms | mutability forces defensive copies; runs EVERY training step |
| hash / eq              | 0.1-0.2 ms | `hash(repr(...))` built from scratch |

Structural causes (full map in the session record): complex types are
lists of `Ty` objects rebuilt on every `@`; `Layer.dom/cod` are
computed properties doing two tensor ops per access; `offsets` is an
uncached property recomputed in snake-removal's innermost loop;
functor cache hits pay a pickle round-trip; normal form does O(n^2)
list-slice interchanges; no `__slots__`/frozen anywhere; validation
(`isinstance`/category scans, dom/cod equality) runs on every
composition operator.

Two root causes: **mutability** (forces copying) and **object-heavy
types** (forces allocation and string hashing). The design removes
both rather than micro-optimising them.

## Decisions (made with user)

1. **New fast core + thin shims** — a new module with the ideal
   representation; the existing `grammar` classes stay untouched and
   interoperate via converters. Not a drop-in rewrite, not a clean
   break.
2. **Pure Python now, Rust kernels later** — the core uses flat
   integer-based structures chosen so that bobcat_rs-style kernels for
   normal-form/rewriting can be added later WITHOUT changing the data
   model, gated on post-redesign profiles.
3. **Validation off by default, on in tests/debug** — correctness
   checks are kept but moved behind a switch; production pipelines
   stop paying per-operation taxes for invariants their inputs already
   satisfy.

## Goals & non-goals

- **Goals:** >=10x on the five measured operations; eliminate
  per-training-step diagram copying entirely; lossless round-trip with
  `grammar.Diagram` for the supported feature set; numerically
  identical end-to-end results.
- **Non-goals (v1):** frames / hierarchical boxes (DisCoCirc) — the
  shim raises cleanly and the old path remains; quantum circuits
  (`backend.quantum`) beyond what the shims give for free; drawing
  (convert to `grammar` and use the existing drawer); replacing the
  old module (it stays as the reference implementation and fallback).

## The core model (`lambeq/backend/fast/`)

### `types.py` — interned atoms, tuple types

- Global intern table: `(name: str, z: int) -> atom id (int)`.
  Adjoints are ordinary ids; `.l`/`.r` are id->id arrays extended
  lazily when new (name, z) pairs are first requested. Parallel arrays
  hold name/z metadata for repr and conversion.
- `FTy`: immutable, `__slots__`, wrapping `atoms: tuple[int, ...]`
  with a stored hash. `@` = tuple concat; rotation = reversed table
  lookup; `len`/indexing/slicing O(1)/O(k). No recursion, no
  per-element isinstance scans, no list materialisation.

### `diagram.py` — term presentation with stored offsets

- `FBox`: immutable, `__slots__`: `name`, `dom: FTy`, `cod: FTy`,
  `kind` (enum: PLAIN/WORD/CUP/CAP/SWAP/SPIDER/DAGGERED), `z`,
  `payload` (data/symbols reference), stored structural hash.
  Equality by fields (ints/tuples).
- `FDiagram`: immutable, `__slots__`: `dom: FTy`,
  `terms: tuple[tuple[FBox, int], ...]` (box, offset), `cod: FTy`
  (computed once at construction), stored hash.
  - `>>`: term concatenation (+ dom/cod check only under validation).
  - `@`: term concatenation with constant integer offset shift
    (`offset + len(left.cod)`).
  - `id`, `cups`, `caps`, `swap`, `spiders`, `dagger`, `rotate`
    constructors mirroring the grammar API surface used by consumers.
  - `offsets`, `boxes` are O(1) views over stored data.
- The `(box, offset)` flat-array form is the Rust-kernel-ready
  representation: future kernels receive integer arrays + a box table.

### `validate.py` — opt-in correctness

- Module switch: `fast.set_validation(bool)`, context manager
  `fast.validation()`, env `LAMBEQ_FAST_VALIDATE=1`. The test suite
  runs with validation ON.
- Checks (all the guarantees the old module enforced on hot paths):
  dom/cod agreement in `then`, adjoint compatibility in cups/caps,
  offset bounds in term construction, type-width consistency.
- Error messages cite term index and offset (better than the current
  generic mismatch errors).

### `functor.py` — copy-free functors

- `FFunctor(ob, ar)`: `ob` as an atom-id -> FTy table with callable
  fallback; `ar` callable with cache keyed by `FBox` (stored-int
  hash). **Cache hits return the shared immutable object** — the
  pickle-per-hit and the double dom/cod re-validation are gone by
  construction (a single validation under the debug switch only).
- Application is a single pass over terms with prefix-sum offset
  arithmetic for width changes — no per-layer `id(left) @ f(box) @
  id(right)` triple-functor calls.

### `normal.py` — rewriting on offset arrays

- Ports of cup/snake removal and interchange/normal form operating
  directly on the terms array (offset arithmetic instead of
  Layer-list slicing; `offsets` reads are O(1)). Same asymptotics as
  today in v1; the flat representation is the prerequisite for Rust
  kernels if post-redesign profiles still show these dominating.

### `contraction.py` — the training-path unlock

- `FDiagram.to_contraction() -> ContractionSpec`: an einsum-style
  spec (indices, sizes) plus an ordered symbol-slot table, extracted
  once per diagram. Training steps gather weights into slots — **zero
  diagram copies or mutations per step** (replaces the
  deepcopy-then-mutate `_fast_subs` pattern). This formalises the
  `true_diagram`/`true_symbols` format already used in discoviz.
- v1 integrates it behind the existing model API for the tensor path
  (`PytorchModel`-equivalent usage); quantum-circuit models keep the
  old path.

### `convert.py` — the shims

- `to_fast(grammar.Diagram) -> FDiagram` and
  `to_grammar(FDiagram) -> grammar.Diagram`: lossless both ways for
  the v1 feature set (plain boxes, words, cups/caps/swaps/spiders,
  daggered). Frames -> `NotImplementedError` with a clear message.
- `CCGTree.to_fast_diagram()`: builds `FDiagram` directly with the
  same recursion as `to_diagram` (skips the construction tax too).
- `tensor.Diagram` (Dim-typed) is handled by the same machinery: Dim
  atoms intern as `(str(dim), 0)`-style entries carrying the integer
  dimension in atom metadata.

## Testing & gates

1. **Unit:** type algebra (tensor/rotation round-trips, interning),
   diagram constructors, validation on/off behaviour, functor caching
   identity semantics.
2. **Round-trip gate:** `to_grammar(to_fast(d)) == d` for the
   300-caption corpus diagrams AND randomized programmatic diagrams
   (cups/swaps/spiders mixes).
3. **Pipeline equivalence gate:** old pipeline (to_diagram ->
   remove_cups -> SpiderAnsatz -> contraction with fixed weights) vs
   fast pipeline produce numerically identical tensors over the
   300-caption corpus.
4. **Existing suite:** untouched modules keep passing; new tests run
   with validation ON.

## Performance targets (same 300-caption benchmark)

| operation | today | target |
|---|---:|---:|
| construction (`to_fast_diagram`) | 2.7 ms | <=0.3 ms |
| remove_cups | 3.9 ms | <=0.4 ms |
| ansatz functor | 7.5 ms | <=0.75 ms |
| copy | 1.4 ms | ~0 (shared reference) |
| per-step substitution | copy+mutate per step | one-off extraction + per-step gather |

Misses are reported honestly with a profile of where the model broke,
as in the parser campaign. Benchmark harness + results land in
`benchmarks/` alongside the Bobcat ones.

## Delivery phases

1. `types.py` + `diagram.py` + `validate.py` + unit tests.
2. `convert.py` shims + round-trip gate + `to_fast_diagram`.
3. `functor.py` + `normal.py` (cup removal, normal form) + pipeline
   equivalence gate.
4. `contraction.py` + tensor-model integration + benchmark suite +
   RESULTS.
