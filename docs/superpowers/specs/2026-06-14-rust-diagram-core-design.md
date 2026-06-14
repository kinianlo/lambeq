# Rust diagram core + construction kernel (stage A) — design

**Date:** 2026-06-14
**Status:** approved
**Context:** This is **stage A** of a larger roadmap to move the
post-parser pipeline into Rust. The full pipeline the user wants in Rust:

```
parse (DONE: bobcat_rs)  ->  A: CCG tree -> diagram (FDiagram)
                          ->  B: ansatz (diagram -> tensor network)
                          ->  C: -> einsum spec (the format training consumes)
```

Stages B and C are separate future specs. Quantum-circuit ansätze
(IQP/etc. -> pennylane/tket) are out of scope for the whole roadmap for
now (tensor/einsum path only). **A is the prerequisite**: it establishes
the shared Rust data model (FTy/FBox/FDiagram + intern table + symbol
slots) that B and C reuse, and it captures the construction win.

### Why now (the profile that gates this)

The Python direct construction (branch `fast-construction`, 3.3x:
2.85 -> 0.76ms/diagram) eliminated the `grammar.Diagram` object churn
(94% of the old cost). What remains is ~0.62ms of per-node Python
assembly compute above the 0.14ms type-resolution floor. Crucially, the
user observed that **construction (0.76ms/diagram) is now slower than the
GPU parser (~0.59ms/sentence at ~1700 sent/s)** — diagram construction
has become the text->diagram bottleneck. Unlike the pre-`fast-construction`
state, the residual is now genuine per-node compute (not object churn),
which Rust speeds up, and construction is serial single-thread Python
CPU, which rayon parallelism fixes. The profile finally points at Rust.

## Goal & non-goals

- **Goal:** A Rust-native diagram data model and a batched, rayon-parallel
  construction kernel that builds `FDiagram`s from Python-resolved CCG
  trees, bit-identical to `to_diagram`, and faster than both the Python
  direct path and the parser throughput. Establish `RsDiagram` as the
  persistent handle stages B/C will consume.
- **Non-goals (A):** porting `_resolved`/`CCGType` to Rust (stays Python
  — it is cheap at 0.14ms and the most intricate type logic; the highest
  port risk for the least gain); the ansatz (stage B); contraction-spec
  extraction (stage C); quantum circuits; fusing construction into the
  parser lane (a later refinement once the data model is proven);
  changing the Python fast core (`lambeq/backend/fast/`) data model.

## Decisions (made with user)

1. **Whole pipeline in Rust is the north star**, decomposed A -> B -> C,
   tensor-path only, quantum deferred. Build A first.
2. **Incremental boundary for A:** Python keeps `_resolved` + the
   `CCGType` accessor extraction (the type logic); Rust does the
   assembly. Defer the full parser-tree fusion until the data model is
   proven.
3. **Batched + rayon-parallel** construction is the headline — it
   amortizes the PyO3 crossing (the real risk for ~15-node trees) and
   parallelizes the serial-Python bottleneck so construction can keep up
   with the GPU parser.
4. **Opt-in** like `parser_backend`: a backend switch defaulting to
   Python until the Rust path is proven.

## Architecture

New modules in the existing `rust/` crate (`bobcat_rs`), so construction
can later fuse with the parser and share the maturin/toolchain build:

- `rust/src/fdiagram.rs` — the data model:
  - A process-global atom **intern table** `(name: String, z: i32) -> u32`
    behind a `OnceCell<Mutex<...>>` (mirrors the Python global), with
    parallel name/z vectors and lazily-built `.l`/`.r` adjoint id arrays.
  - `FTy` = `Vec<u32>` (atom ids) with tensor (concat), `.l`/`.r`
    (reversed adjoint lookup), slicing.
  - `FBox` = `{ name, dom: FTy, cod: FTy, kind: u8, z: i32, is_dagger:
    bool }` (payload is None for stage A — construction produces only
    structural boxes + word boxes whose "payload" is the word name).
  - `FDiagram` = `{ dom: FTy, terms: Vec<(FBox, u32)>, cod: FTy }` with
    an optional frontier-replay validation (debug builds / a flag).
- `rust/src/build.rs` — the combinator fragments and assembly:
  - `fa/ba/fc/bc/fx/bx/gfc/gbc/gfx/gbx/ftr/btr/swaps` over Rust `FTy`,
    mirroring `lambeq/backend/fast/build.py` (which is already gated vs
    the grammar oracle) — same formulas, native arrays.
  - `assemble(program) -> FDiagram`: executes a build program — a
    post-order node list, each `(rule_tag, [arg FTys], word_name?)` —
    doing the LEXICAL/UNARY/binary cases of `_to_fast_diagram`
    (tensor children, then `>> rule_fragment`).
- `rust/src/lib.rs` (extend) — PyO3 surface:
  - `RsDiagram` PyO3 class wrapping `FDiagram`, with `.to_fast()` ->
    Python `lambeq.backend.fast.FDiagram`. **The Rust and Python intern
    tables are independent**: `.to_fast()` maps each Rust atom id back to
    its `(name, z)` and re-interns through the Python `atom(name, z)`
    function, so the materialised Python `FTy`/`FBox` carry *Python* atom
    ids (never raw Rust ids). Built via the existing Python
    `FDiagram(FTy(...), terms, FTy(...))` / `FBox(...)` constructors so
    Python-side hashing/validation is automatic.
  - `build_diagrams(programs) -> list[RsDiagram]` — the batched entry
    point; rayon-parallel across programs.

Python side:
- `lambeq/text2diagram/ccg_tree.py` — a build-program *emitter*. Given a
  resolved tree, walk it and produce the serialized program (reusing the
  exact accessor logic of `_fast_rule_layer`/`_to_fast_diagram`, but
  emitting `(rule_tag, arg-atom-lists, text)` tuples instead of building
  FDiagrams). A `to_fast_diagram(backend='python'|'rust')` switch:
  `'rust'` emits the program, calls `bobcat_rs.build_diagrams([program])`,
  returns `rs[0].to_fast()`. A batch helper
  `trees_to_fast_diagrams(trees, backend=...)` emits all programs and
  makes ONE Rust call.

## Data flow

```
CCGTree --collapse+_resolved (Py)--> resolved tree
        --emit build-program (Py)---> program (rule tags + (name,z) atom lists)
   [batch all programs]
        --bobcat_rs.build_diagrams (Rust, rayon)--> [RsDiagram]
        --.to_fast() (per diagram)--> Python FDiagram   # gate + drop-in compat
        --[stage B/C later consume RsDiagram natively]
```

The Python/torch numeric tail is unchanged: `RsDiagram.to_fast()` yields
the same `FDiagram`, and the existing functor/contraction/torch path runs
on it identically. Stage A is purely the structural builder.

## The build-program format (the boundary contract)

A program is a flat post-order list of nodes. Each node is one of:
- **Word:** `(LEXICAL, cod_atoms, text)` — `cod_atoms` is a list of
  `(name, z)`. Punctuation leaves emit `(LEXICAL_PUNC,)` (empty).
- **Unary-swap:** `(UNARY, right_atoms, left_atoms)` — the swap arg types.
- **Binary:** `(rule_tag, [arg_atom_lists...])` — the pregroup-type args
  that rule's fragment needs, in the order `build.<rule>` expects (e.g.
  FA: `[left.result, right]`; GFX: `[mid, l, join, r]`). The Python
  emitter computes these via the same `CCGType` accessors as
  `_fast_rule_layer`; Rust just interns and runs `build.<rule>`.

Node arity (0 for leaves, 2 for binary, 1 for unary) lets Rust rebuild
the tree shape for the bottom-up tensor/compose without sending child
pointers — a post-order stack machine.

## Testing & gates

1. **Rust unit tests** (`cargo test`): intern table round-trips, FTy
   tensor/adjoint, each build fragment's shape, the stack-machine
   assembly on a hand-built program.
2. **Differential gate (Python):** `to_grammar(tree.to_fast_diagram(
   backend='rust').to_fast()) == tree.to_diagram()` over the full corpus
   (the same oracle the Python path passes). 0 mismatches required.
3. **Rust ≡ Python gate:** `tree.to_fast_diagram(backend='rust').to_fast()
   == tree.to_fast_diagram(backend='python')` per corpus diagram (the two
   builders agree exactly, including stored hashes).
4. **Existing suites stay green** — the Python path is untouched; the
   Rust path is opt-in.

## Performance

Benchmark batched construction over the corpus: Python-serial direct
(0.76ms/diagram baseline) vs Rust batched single-thread vs Rust batched
rayon (all cores). Report per-diagram ms and total throughput
(diagrams/s), and compare against parser throughput (~1700 sent/s).
Targets: single-thread Rust beats 0.76ms; rayon throughput exceeds the
parser so construction stops being the bottleneck. Honest profile if the
PyO3 marshalling dominates for small diagrams (mitigated by batching;
report the crossover batch size).

## Build & integration

- `pip install ./rust` rebuilds `bobcat_rs` (maturin; rustup toolchains
  on the project share per [[lambeq-dev-environment]]). The new symbols
  (`RsDiagram`, `build_diagrams`) join the existing parser exports.
- `LAMBEQ_FAST_CONSTRUCT_BACKEND` env override + the explicit `backend=`
  kwarg, defaulting to `'python'`. The Rust path is opt-in until gates 2
  and 3 pass over the full corpus, then can become the default in a
  follow-up.

## Risks

- **PyO3 marshalling per small diagram** eating the assembly saving —
  mitigated by the batched `build_diagrams` call; measure the crossover.
- **Intern-table thread-safety** under rayon — the global table needs a
  `Mutex` (or sharded/lock-free map); interning contention could limit
  parallel scaling. Measure; if contention bites, pre-intern the batch's
  atoms single-threaded then build lock-free.
- **Two builders to keep in sync** (Python `build.py` + Rust `build.rs`)
  — gate 3 (Rust ≡ Python) pins them together so drift is caught.

## Delivery phases (this spec -> one plan)

1. `fdiagram.rs` data model + `cargo test`.
2. `build.rs` fragments + assembly stack machine + `cargo test`.
3. PyO3 `RsDiagram` + `build_diagrams`; Python emitter +
   `to_fast_diagram(backend='rust')`; gates 2 & 3 over the corpus.
4. Benchmark (batched, single vs rayon) + RESULTS + push.
