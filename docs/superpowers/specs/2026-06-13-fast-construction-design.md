# Direct CCGTree → FDiagram construction — design

**Date:** 2026-06-13
**Status:** approved
**Context:** The fast diagram core (`lambeq/backend/fast/`) is merged to
main. Construction is the one operation that did NOT beat its target:
`CCGTree.to_fast_diagram()` is still the thin wrapper
`convert.to_fast(self.to_diagram())`, so it pays the full cost of
building a `grammar.Diagram` and then converting it. A profile (299 COCO
diagrams, i7-11800H) localised the cost precisely:

| stage | ms/diagram | share |
|---|---:|---:|
| `collapse_noun_phrases` + `_resolved` (CCG type logic) | 0.140 | 6% |
| diagram assembly (`grammar.Diagram`/`Layer`/`Cup` objects) | 2.064 | 94% |
| `convert.to_fast` (the fast-core part) | 0.163 | — |
| **`to_fast_diagram` total** | **~2.39** | |

94% of construction is `grammar.Diagram` object churn — the mutability,
per-op validation, and repr-hashing the fast core was built to avoid. A
Rust port was considered and rejected: the Rust-worthy compute (the type
logic) is only 0.14ms, while the bottleneck (object allocation) is
eliminated in pure Python by never building a `grammar.Diagram`. Rust
would add an FFI boundary and a CCG-rule reimplementation for ~0.1-0.2ms
marginal gain — the wrong tool here.

## Goal & non-goals

- **Goal:** Rewrite `CCGTree.to_fast_diagram()` to build the `FDiagram`
  term array directly from the resolved CCG tree, never allocating a
  `grammar.Diagram`. Target ~2.4ms → <=0.4ms construction. Output must
  remain bit-identical to today's path (the existing differential gate
  must stay green and becomes non-trivial).
- **Non-goals:** `to_diagram` stays the untouched reference/oracle; only
  the `planar=False` derivation `to_fast_diagram` already uses is
  supported (planar=True is not exposed by `to_fast_diagram`); frames
  still raise `NotImplementedError`; no Rust; no change to the fast
  core's data model or to `_resolved`/`collapse_noun_phrases`.

## Decisions (made with user)

1. **Pure Python, no Rust** — the bottleneck is object churn, killed by
   emitting `(FBox, offset)` terms directly; the 0.14ms type logic does
   not justify a Rust port or its FFI boundary.
2. **Reuse the existing type logic** — `collapse_noun_phrases()` and
   `_resolved()` are correct and cheap; the rewrite only replaces the
   bottom-up *assembly* (`_to_diagram`), not the top-down type pass.
3. **Reuse `CCGType.to_grammar() -> ty_to_fast` for per-node types** —
   not worth a separate CCGType→FTy path for 0.14ms; the win is in
   assembly, not type mapping.
4. **Builder helpers in a new `lambeq/backend/fast/build.py`** — keeps
   `diagram.py` focused on the data model.

## Architecture & data flow

`CCGTree.to_fast_diagram()` becomes:

```
resolved = self.collapse_noun_phrases()._resolved()
words, grammar = resolved._to_fast_diagram()   # new bottom-up recursion
return words >> grammar
```

`_to_fast_diagram()` mirrors `_to_diagram(planar=False)` exactly
(`ccg_tree.py:458`), returning a pair of `FDiagram`s:
- `words`: dom `FTy()`, cod = tensor of all word output types.
- `grammar`: dom = `words.cod`, cod = derivation result type.

Recursion:
- **LEXICAL leaf:** punctuation → `(FDiagram.id(), FDiagram.id())`
  (no box, vanishes under tensor); otherwise
  `(word(text, cod).to_diagram(), FDiagram.id(cod))` where
  `cod = ty_to_fast(biclosed_type.to_grammar())`.
- **Binary node:** `words = lw @ rw`; `grammar = (lg @ rg) >> rule_layer(rule, child_ftypes, output_ftype)`.
- **UNARY (direction-change swap only):** `grammar = childg >> swap_layer`
  (the `swap(right, left)` case at `ccg_tree.py:469-483`).

### `lambeq/backend/fast/build.py`

Fast-core equivalents of `grammar.Diagram`'s CCG combinators, each taking
`FTy` arguments and returning an `FDiagram`, following the formulas
mapped from `ccg_rule.py` (the `code-explorer` reference, with exact line
refs preserved in the plan):

- `fa(X, Y)` = `Id(X) @ cups(Y.l, Y)`
- `ba(Y, X)` = `cups(Y, Y.r) @ Id(X)`
- `fc(X, Y, Z)` = `Id(X) @ cups(Y.l, Y) @ Id(Z.l)`
- `bc(Z, Y, X)` = `Id(Z.r) @ cups(Y, Y.r) @ Id(X)`
- `fx(X, Y, Z)` = two layers: `(Id(X) @ swap(Y.l, Z.r) @ Id(Y)) >> (swap(X, Z.r) @ cups(Y.l, Y))`
- `bx(Z, Y, X)` = `(Id(Y) @ swap(Z.l, Y.r) @ Id(X)) >> (cups(Y, Y.r) @ swap(Z.l, X))`
- `gfc`, `gbc` = generalised cup-sequences with identity tails
- `gfx`, `gbx` = split-based swap+cups (use the `CCGType.split` result,
  mapped to `FTy`)
- type-raising fragments: `ftr(T, A)` = `caps(T, T.l) @ Id(A)`;
  `btr(T, A)` = `Id(A) @ caps(T.r, T)`
- `rule_layer(rule, dom_ftypes, output_ftype) -> FDiagram` — the
  dispatcher mirroring `CCGRule.apply` (`ccg_rule.py:176`), including
  LP/RP (identity on the surviving child) and the type-raising/unary
  cases.

These reuse the fast core's existing `cups`, `caps`, `word`,
`FDiagram.id`, `>>`, `@`.

**Swap decomposition (correctness-critical).** `build.py` must NOT use a
single complex `SWAP` FBox for multi-atom swaps. `grammar.Diagram.swap`
expands a complex swap into elementary atomic swaps at construction
(`Swap.__new__`, `grammar.py:1853`), so `to_diagram`'s output — and hence
the differential oracle — contains the *expanded* form, and the existing
`convert.to_fast` only ever encounters already-expanded atomic swaps.
`build.py` therefore needs a decomposing `swaps(left, right) -> FDiagram`
helper that mirrors `Swap.__new__`'s bubble-sort expansion (for each atom
`ob` in `right`, bubble it left past each atom of `left`:
`swap(left[i], ob)` at offset `start + i`, `i` descending). The
single-box `fast.diagram.swap` is correct only for the atomic case and is
used as the per-elementary-swap building block. Using a single complex
SWAP box would also break `box_to_grammar`, which rebuilds a
`grammar.Swap` (itself a multi-layer diagram for complex types), not a
single box.

## Testing & gates

1. **Differential gate (the safety net):** the existing
   `tests/backend/test_fast_convert.py::test_to_fast_diagram_equivalent`
   asserts `to_grammar(t.to_fast_diagram()) == t.to_diagram()`. Today it
   is trivially true (wrapper); after the rewrite it is a real
   differential test against the `to_diagram` oracle. It runs with
   validation ON (conftest). KEEP it; it is the gate every rule must
   pass.
2. **Per-rule unit tests:** small hand-built `CCGTree`s exercising each
   rule case (FA, BA, FC, BC, FX, BX, GFC, GBC, GFX, GBX, FTR, BTR,
   LP, RP, unary-swap), each asserting fast == oracle. These guarantee
   coverage the corpus may not provide for rarer rules.
3. **Full-corpus gate:** the benchmark asserts fast == oracle over ALL
   parsed corpus diagrams (not just the 40-sentence test fixture).
4. **Existing suites stay green** — `to_diagram` and the rest of the
   fast core are untouched.

## Performance reporting

`benchmarks/fastdiag_bench.py` already times `to_diagram` vs
`to_fast_diagram` in its construction row; after the rewrite it shows the
real speedup. Append a short note to `benchmarks/RESULTS.md` with the new
construction number and the speedup vs the old wrapper, plus the
resolved/assembly split that motivated the change. Honest verdict if the
0.4ms target is missed, with a one-line profile.

## Delivery

Single plan on branch `fast-construction` (forked from main). TDD,
rule-by-rule, each gated against the `to_diagram` oracle. The only
library files touched: new `lambeq/backend/fast/build.py`,
`lambeq/text2diagram/ccg_tree.py` (`to_fast_diagram` + `_to_fast_diagram`),
and re-exports in `lambeq/backend/fast/__init__.py`. `to_diagram` is not
modified.
