# Fast box-level Rewriter (`FRewriter`) — design

**Date:** 2026-06-17
**Status:** approved (user pre-approved the plan series; build autonomously)
**Context:** lambeq's box-level `Rewriter` (rewrite/base.py:363) applies a
set of `RewriteRule`s (determiner, connector, coordination, curry,
prepositional_phrase, etc.) to a diagram via a `grammar.Functor`
(ob=identity, ar = first matching rule's rewrite, else the box). It is a
standard preprocessing step and, like `SpiderAnsatz`, is
`grammar.Functor`-based → it pays the `grammar.Diagram` object churn the
fast core eliminates. None of lambeq's rewriters are fast-compatible
beyond the already-ported `remove_cups`/snake-removal. This ports the
box-level `Rewriter` onto the copy-free `FFunctor`.

## Goal & non-goals

- **Goal:** `FRewriter`, a fast-core equivalent of `Rewriter` operating on
  `FDiagram` via `FFunctor`, numerically/structurally identical to the
  legacy `Rewriter` (gated: `to_grammar(FRewriter(rules)(to_fast(d)))
  == Rewriter(rules)(d)` over the corpus), capturing the object-churn
  win (~the FFunctor 100×-class speedup on the traversal).
- **Non-goals:** porting individual `RewriteRule`s to fast (we REUSE the
  legacy rules' `matches`/`rewrite` per box — they're diverse and
  correct; only the traversal goes fast); the whole-diagram
  `DiagramRewriter`s (`RemoveSwapsRewriter`, `UnifyCodomainRewriter` —
  separate term-walk ports like `remove_cups`, future work); quantum;
  changing `Rewriter`/`RewriteRule` (they stay as oracle + delegate).

## Design (reuse legacy rules, fast traversal)

`FRewriter(rules=None)` mirrors `Rewriter`:
- Rule resolution: reuse the legacy `Rewriter`'s rule resolution — hold a
  legacy `Rewriter(rules)` internally and take its resolved
  `.rules: list[RewriteRule]` (so `FRewriter` accepts the same
  `Iterable[RewriteRule | str] | None`, presets and all).
- `self.functor = FFunctor(ob=self._ob, ar=self._ar)`.
- `_ob(functor, atom_id) -> FTy`: identity — `FTy((atom_id,))` (the
  legacy `Rewriter._ob` is identity on types; rewrites preserve the
  box interface, so types are unchanged).
- `_ar(functor, fbox)`: reconstruct a `grammar.Box`/`Word` from the FBox
  (`name`, `convert.ty_to_grammar(dom)`, `ty_to_grammar(cod)`, `z`,
  daggered/word per `fbox.kind`/`is_dagger`); for `rule in self.rules`:
  `out = rule(gbox)` (legacy `RewriteRule.__call__` = rewrite if matches
  else None); if `out is not None`, return
  `convert.to_fast(out.to_diagram() if not isinstance(out, grammar.Diagram)
  else out)` (the rewritten fragment, fast); else (no rule matched)
  return the `fbox` unchanged. First matching rule wins (mirrors
  `Rewriter._ar`).
- `__call__(fd: FDiagram) -> FDiagram` = `self.functor(fd)`.

**Why this captures the win:** the expensive part of the legacy
`Rewriter` is the whole-diagram `grammar.Functor` application (object
allocation + validation + repr-hash per layer). `FRewriter` runs that
traversal as the copy-free `FFunctor`; the per-box rule
`matches`/`rewrite` (small grammar fragments, only for *matched* boxes —
most boxes match nothing and pass through) reuses the legacy logic
unchanged. Same reuse pattern as the ansatz's uncurry fallback.

**Interface preservation:** a rewrite fragment has the same dom/cod as
the box it replaces (e.g. determiner `Word('the', N@N.l)` →
`Cap(N, N.l)`, both dom `Ty()`/cod `N@N.l`), so `to_fast(fragment)` has
the same `FTy` dom/cod as the original FBox and `FFunctor` splices it by
offset. Types are unchanged (ob=identity), consistent with `Rewriter`.

## Testing & gates

1. **Corpus differential (headline):** for the default rules AND an
   extended set (incl. `coordination`, `curry`),
   `to_grammar(FRewriter(rules)(to_fast(d))) == Rewriter(rules)(d)` over
   the corpus. 0 mismatches. (This exercises the rules that actually fire
   on COCO; report which rules fire.)
2. **Per-rule unit checks:** small hand-built diagrams that trigger
   specific rules (determiner, connector) → `FRewriter([rule]) ==
   Rewriter([rule])`.
3. **Composability:** `FRewriter` output feeds `FSpiderAnsatz` /
   `to_contraction` (a rewritten diagram still contracts) — a quick
   numeric check vs the legacy `Rewriter → SpiderAnsatz → PytorchModel`.
4. **Existing suites stay green** (additive; legacy untouched).

## Performance

`benchmarks/fastrewriter_bench.py`: legacy `Rewriter(rules)(d)` vs
`FRewriter(rules)(to_fast(d))` over the corpus, ms/diagram + speedup.
Pre-assert the differential before timing. Append to `RESULTS.md`. Honest
note: a once-per-dataset preprocessing win; the per-box matched-rule
fragments still pay small grammar cost (reuse), so the speedup is bounded
by how many boxes match (most don't → mostly pure FFunctor traversal).

## Delivery (one plan)

Branch `fast-rewriter` (from `fast-pipeline`). TDD: `FRewriter` + per-rule
+ corpus differential gate; then composability check; then benchmark.
Files: new `lambeq/backend/fast/rewrite.py`, `__init__.py` re-export,
`tests/backend/test_fast_rewriter.py`, `benchmarks/fastrewriter_bench.py`,
`RESULTS.md`. Finish: merge into `fast-pipeline` (like the quantum
front-end).
