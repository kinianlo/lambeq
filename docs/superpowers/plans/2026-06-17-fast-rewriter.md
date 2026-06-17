# Fast box-level Rewriter (FRewriter) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `FRewriter`, a fast-core equivalent of lambeq's box-level `Rewriter` operating on `FDiagram` via `FFunctor`, structurally identical to the legacy `Rewriter`, capturing the object-churn win.

**Architecture:** `FFunctor(ob=identity, ar=_ar)`; `_ar` reconstructs a `grammar.Box` from each FBox, delegates to the legacy `RewriteRule.__call__` (reuse all rule logic), converts the matched fragment with `to_fast`, else returns the FBox unchanged. Rule resolution reuses a legacy `Rewriter` internally.

**Tech Stack:** Pure Python. Oracle: `lambeq.rewrite.Rewriter` (base.py:363) + its `RewriteRule`s. Fast core: `FFunctor`, `convert.{to_fast,to_grammar,ty_to_grammar}`, `fast.types.{atom_name,atom_z,FTy}`.

**Spec:** `docs/superpowers/specs/2026-06-17-fast-rewriter-design.md`

---

## Conventions

- Repo `/home/kinianlo/projects/lambeq`, branch `fast-rewriter` (off
  `fast-pipeline`). `PY=~/.pyenv/versions/qnlp/bin/python`. 600000ms
  timeouts on Bobcat-loading commands. Corpus `/tmp/coco_bench.txt`.
- Commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Additive: `lambeq/rewrite/` and `lambeq/ansatz/` UNCHANGED (oracles).

## Verified facts

- `Rewriter(rules=None)` (base.py:363): `rules` is `Iterable[RewriteRule
  | str] | None`; resolves to `self.rules: list[RewriteRule]`;
  `apply_rewrites = Functor(grammar, ob=identity, ar=_ar)`; `_ar(box)`:
  first `rule(box)` that returns non-None wins, else `box`; `__call__(d)
  = apply_rewrites(d)`. `RewriteRule.__call__(box) = rewrite(box) if
  matches(box) else None`. Default rules: auxiliary, connector,
  determiner, postadverb, preadverb, prepositional_phrase. Also
  available: coordination, curry, object_rel_pronoun, subject_rel_pronoun.
- `FFunctor(ob, ar)`: `ob(functor, atom_id) -> FTy`; `ar(functor, fbox)
  -> FBox | FDiagram` (returning an FDiagram splices it); structural
  boxes auto-mapped (don't hit ar). `FBox`: `name,dom,cod,kind,z,
  is_dagger,payload`; kinds WORD/PLAIN/... from fast.diagram.
- `convert.ty_to_grammar(FTy)->grammar.Ty`, `convert.to_fast(grammar
  .Diagram)->FDiagram`, `convert.to_grammar(FDiagram)->grammar.Diagram`.
- rewrite fragments preserve box dom/cod (interface), so the FFunctor
  splice is offset-valid; ob is identity (types unchanged).

## File map
```
lambeq/backend/fast/rewrite.py     FRewriter
lambeq/backend/fast/__init__.py    + re-export FRewriter
tests/backend/test_fast_rewriter.py
benchmarks/fastrewriter_bench.py
benchmarks/RESULTS.md
```

---

### Task 1: `FRewriter` (`lambeq/backend/fast/rewrite.py`) + gates

**Files:** Create `lambeq/backend/fast/rewrite.py`; modify
`lambeq/backend/fast/__init__.py`; Test
`tests/backend/test_fast_rewriter.py`.

FIRST read `lambeq/backend/fast/functor.py` to confirm `ob`/`ar`
signatures + that structural boxes bypass `ar` (so `_ar` only sees
PLAIN/WORD), and `lambeq/backend/fast/ansatz.py` for the gbox-
reconstruction pattern (it reconstructs grammar boxes from FBoxes for the
legacy `_summarise_box`).

- [ ] **Step 1: implement `lambeq/backend/fast/rewrite.py`:**

```python
# (Apache header — copy from lambeq/backend/grammar.py)
"""FFunctor-based box-level Rewriter on the fast diagram core."""

from __future__ import annotations

from collections.abc import Iterable

from lambeq.backend import grammar
from lambeq.backend.fast import convert
from lambeq.backend.fast.diagram import FBox, FDiagram, WORD
from lambeq.backend.fast.functor import FFunctor
from lambeq.backend.fast.types import FTy
from lambeq.rewrite.base import Rewriter, RewriteRule


class FRewriter:
    """Fast-core equivalent of :class:`lambeq.rewrite.Rewriter`.

    Same rules/API; runs the diagram traversal as a copy-free FFunctor.
    Per box, delegates to the legacy ``RewriteRule.__call__`` (rewrite if
    matches else None) on a reconstructed grammar box, converting a
    matched fragment back to the fast core. Identical output to the
    legacy ``Rewriter``; faster (no grammar.Diagram object churn on the
    traversal).
    """

    def __init__(self,
                 rules: Iterable[RewriteRule | str] | None = None) -> None:
        # reuse the legacy Rewriter's rule resolution (presets etc.)
        self.rules = Rewriter(rules).rules
        self.functor = FFunctor(ob=self._ob, ar=self._ar)

    def __call__(self, diagram: FDiagram) -> FDiagram:
        return self.functor(diagram)

    def _ob(self, functor, atom_id: int) -> FTy:
        return FTy((atom_id,))     # identity (rewrites preserve types)

    def _ar(self, functor, box: FBox):
        gbox = self._to_grammar_box(box)
        for rule in self.rules:
            out = rule(gbox)
            if out is not None:
                diag = out if isinstance(out, grammar.Diagram) \
                    else out.to_diagram()
                return convert.to_fast(diag)
        return box     # no rule matched: unchanged

    def _to_grammar_box(self, box: FBox) -> grammar.Box:
        dom = convert.ty_to_grammar(box.dom)
        cod = convert.ty_to_grammar(box.cod)
        if box.kind == WORD:
            gb = grammar.Word(box.name, cod)
        else:
            gb = grammar.Box(box.name, dom, cod, z=box.z)
        return gb.dagger() if box.is_dagger else gb
```

NOTES for the implementer:
- Confirm the `grammar.Word`/`grammar.Box` constructor signatures and
  that a daggered box is reconstructed faithfully (mirror
  `convert.box_to_grammar`'s inverse — read it; reuse its logic if it
  exposes a single-box helper). The corpus differential gate is the
  arbiter: if a reconstructed box makes a rule mis-fire vs the legacy,
  fix the reconstruction to match what the legacy functor's `ar`
  receives.
- A rewrite fragment's dom/cod equals the box's, so the FFunctor splice
  is valid; if validation (ON in tests) rejects a splice, the
  reconstruction or fragment is wrong — debug against the oracle, don't
  disable validation.
- Re-export `FRewriter` from `lambeq/backend/fast/__init__.py` (import +
  `__all__`).

- [ ] **Step 2: tests** (`tests/backend/test_fast_rewriter.py`):

```python
import pytest

from lambeq import RemoveCupsRewriter
from lambeq.backend import grammar
from lambeq.backend.fast import FRewriter, convert
from lambeq.rewrite import Rewriter


def test_default_rules_match_oracle(bobcat_diagrams):
    assert bobcat_diagrams
    rw = Rewriter()                       # default rules
    fr = FRewriter()
    for d in bobcat_diagrams:
        assert convert.to_grammar(fr(convert.to_fast(d))) == rw(d), d


def test_extended_rules_match_oracle(bobcat_diagrams):
    names = ['determiner', 'connector', 'coordination', 'curry',
             'prepositional_phrase', 'auxiliary',
             'object_rel_pronoun', 'subject_rel_pronoun']
    rw = Rewriter(names)
    fr = FRewriter(names)
    for d in bobcat_diagrams:
        assert convert.to_grammar(fr(convert.to_fast(d))) == rw(d), d


def test_determiner_rule_hand_built():
    n = grammar.Ty('n')
    the = grammar.Word('the', n @ n.l)
    d = (the @ grammar.Word('cat', n)).to_diagram() if False else \
        grammar.Diagram.create_pregroup_diagram(
            words=[grammar.Word('the', n @ n.l), grammar.Word('cat', n)],
            morphisms=[(grammar.Cup, 1, 2)])
    rw = Rewriter(['determiner'])
    fr = FRewriter(['determiner'])
    assert convert.to_grammar(fr(convert.to_fast(d))) == rw(d)
```

(Adjust the hand-built determiner case to whatever cleanly triggers the
determiner rule — the substance is `FRewriter([rule]) == Rewriter([rule])`
on a diagram that fires the rule. If `create_pregroup_diagram` types need
tweaking, fix the construction, keep the assertion.)

- [ ] **Step 3: run + flake8 + commit.** `$PY -m pytest
tests/backend/test_fast_rewriter.py -q` (600000ms) → all pass. The corpus
differentials are THE gate; report which rules actually fired on the
corpus (instrument if 0 fired — then the gate is trivially true and you
must add a hand-built case per rule that DOES fire, so the port is really
exercised). `$PY -m flake8 lambeq/backend/fast/rewrite.py tests/backend/test_fast_rewriter.py`.

```bash
git add lambeq/backend/fast/rewrite.py lambeq/backend/fast/__init__.py tests/backend/test_fast_rewriter.py
git commit -m "Add FFunctor-based FRewriter mirroring the box-level Rewriter

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

IMPORTANT on coverage: COCO diagrams may trigger few/no rewrite rules
(the default rules target determiners, connectors, adverbs, etc. which
may be rare post-Bobcat). If the corpus differential passes only because
no rule fires (vacuous), it does NOT validate the port. So: instrument
how many boxes get rewritten across the corpus; if low, ADD hand-built
diagrams that fire each supported rule (determiner, connector,
coordination, curry, prepositional_phrase) and gate `FRewriter([rule]) ==
Rewriter([rule])` on each. Report the fired-rule counts.

---

### Task 2: composability check + benchmark + RESULTS + finish

**Files:** add a test to `tests/backend/test_fast_rewriter.py`; create
`benchmarks/fastrewriter_bench.py`; modify `benchmarks/RESULTS.md`.

- [ ] **Step 1: composability numeric check** (add to
test_fast_rewriter.py) — a rewritten diagram still contracts identically:

```python
def test_rewriter_then_pipeline_numeric(bobcat_diagrams):
    torch = pytest.importorskip('torch')
    from lambeq import AtomicType, PytorchModel, SpiderAnsatz
    from lambeq.backend.fast import FSpiderAnsatz, contraction
    from lambeq.backend.fast.model import FastPytorchModel
    from lambeq.backend.tensor import Dim
    from collections import Counter

    ob = {t: Dim(2) for t in AtomicType}
    rw, fr = Rewriter(), FRewriter()
    rc = RemoveCupsRewriter()
    g = [SpiderAnsatz(ob)(rc(rw(d))) for d in bobcat_diagrams]
    shapes = [tuple(c.cod.dim) for c in g]
    modal = Counter(shapes).most_common(1)[0][0]
    keep = [i for i, s in enumerate(shapes) if s == modal]
    diags = [bobcat_diagrams[i] for i in keep]
    g = [g[i] for i in keep]
    assert len(keep) >= 4
    m = PytorchModel.from_diagrams(g); torch.manual_seed(0)
    m.initialise_weights(); exp = m.get_diagram_output(g)
    W = dict(zip(m.symbols, m.weights))
    fans = FSpiderAnsatz(ob)
    from lambeq.backend.fast.normal import remove_cups
    for d, e in zip(diags, exp):
        fd = fans(remove_cups(fr(convert.to_fast(d))))
        got = contraction.evaluate(contraction.to_contraction(fd), W)
        assert torch.allclose(got, e, atol=1e-5), d
```

(The fast pipeline `FRewriter → remove_cups → FSpiderAnsatz →
to_contraction` matches the legacy `Rewriter → RemoveCups → SpiderAnsatz
→ PytorchModel`. Confirms FRewriter composes into the fast pipeline.)

- [ ] **Step 2: write `benchmarks/fastrewriter_bench.py`** — parse corpus
(rust), time `[Rewriter(rules)(d) for d in diagrams]` vs `[FRewriter(rules)
(convert.to_fast(d)) for d in diagrams]` (or pre-convert the fast inputs
and time the FRewriter call), best-of-3, warm-up, argparse + `--num`.
Pre-assert the differential. Report ms/diagram + speedup + how many boxes
rewritten. Structure like `benchmarks/fastansatz_bench.py`.

- [ ] **Step 3: run** `$PY benchmarks/fastrewriter_bench.py
/tmp/coco_bench.txt --num 400` (note count). Capture numbers.

- [ ] **Step 4: append `## Fast Rewriter` to `benchmarks/RESULTS.md`** —
legacy vs fast ms/diagram + speedup; how many rules/boxes fired; honest
note (preprocessing win; speedup bounded by matched-box count since
matched fragments reuse legacy rule logic). PASTE REAL NUMBERS.

- [ ] **Step 5: full sweep + commit + push**

```bash
$PY -m pytest tests/backend/test_fast_rewriter.py tests/backend/test_fast_ansatz.py tests/backend/test_fast_contraction.py -q
$PY -m flake8 lambeq/backend/fast/rewrite.py benchmarks/fastrewriter_bench.py
git add benchmarks/fastrewriter_bench.py benchmarks/RESULTS.md tests/backend/test_fast_rewriter.py
git commit -m "Benchmark FRewriter and verify pipeline composability

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin fast-rewriter
```

- [ ] **Step 6: report** — corpus differential result + fired-rule
counts, composability gate, speedup, whether push succeeded.
