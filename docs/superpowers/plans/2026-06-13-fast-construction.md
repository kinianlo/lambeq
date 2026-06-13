# Direct CCGTree → FDiagram Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `CCGTree.to_fast_diagram()` to build the `FDiagram` term array directly from the resolved CCG tree (no `grammar.Diagram` allocation), cutting construction from ~2.4ms to <=0.4ms while staying bit-identical to `to_diagram`.

**Architecture:** Reuse the cheap, correct type logic (`collapse_noun_phrases` + `_resolved`, 0.14ms) untouched. Add `lambeq/backend/fast/build.py` with fast-core equivalents of the grammar CCG combinators (FTy in, FDiagram out). Rewrite the bottom-up assembly as `_to_fast_diagram`, mirroring `_to_diagram(planar=False)` but emitting `(FBox, offset)` terms. The CCG-rule dispatch (transcribed from `CCGRule.apply`) lives in `ccg_tree.py` so `backend.fast` stays free of any `text2diagram` dependency.

**Tech Stack:** Pure Python. Oracles: `grammar.Diagram.{fa,ba,fc,bc,fx,bx,cups,caps,swap}` (grammar.py:769-811), `CCGRule.apply` (ccg_rule.py:176-314), `CCGTree._to_diagram` (ccg_tree.py:458+). Gate: the existing differential test, now non-trivial.

**Spec:** `docs/superpowers/specs/2026-06-13-fast-construction-design.md`

---

## Conventions (every task)

- Repo `/home/kinianlo/projects/lambeq`, branch `fast-construction`.
  `PY=~/.pyenv/versions/qnlp/bin/python`. 600000ms timeouts on anything
  loading the Bobcat model. Corpus `/tmp/coco_bench.txt` (2000 captions).
- Commit style: plain sentence + trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- `to_diagram` and the rest of the fast core are NOT modified. Library
  files touched: new `lambeq/backend/fast/build.py`,
  `lambeq/backend/fast/__init__.py` (re-export), and
  `lambeq/text2diagram/ccg_tree.py` (`to_fast_diagram` + new helpers).
- Tests run with validation ON (the autouse fixture in
  `tests/backend/conftest.py`), so every `FDiagram` is frontier-checked
  at construction.

## Fast-core API the builder uses (verified)

- `FTy` (`lambeq/backend/fast/types.py`): `.atoms` tuple, `@`, `.l`, `.r`,
  `len()`, slicing `ty[i:j] -> FTy`, `ty[i] -> int`.
- `lambeq/backend/fast/diagram.py`: `FDiagram.id(ty) -> FDiagram`;
  `cups(left, right) -> FDiagram` (validates `left.r == right`, nested
  innermost-first); `caps(left, right) -> FDiagram`;
  `swap(left, right) -> FBox` (single SWAP box, atomic use only);
  `word(name, cod) -> FBox`; `FDiagram.then`/`>>`, `tensor`/`@`.
- `lambeq/backend/fast/convert.py`: `ty_to_fast(grammar.Ty) -> FTy`.

## Oracle formulas (transcribe exactly)

`grammar.Diagram` combinators (grammar.py:769-792):
```
fa(L, R)      = id(L) @ cups(R.l, R)
ba(L, R)      = cups(L, L.r) @ id(R)
fc(L, M, R)   = id(L) @ cups(M.l, M) @ id(R.l)
bc(L, M, R)   = id(L.r) @ cups(M, M.r) @ id(R)
fx(L, M, R)   = (id(L) @ swap(M.l, R.r) @ id(M)) >> (swap(L, R.r) @ cups(M.l, M))
bx(L, M, R)   = (id(M) @ swap(L.l, M.r) @ id(R)) >> (cups(M, M.r) @ swap(L.l, R))
```
`CCGRule.apply` dispatch (ccg_rule.py:204-314) — `to_grammar()` becomes
`ty_to_fast(...)` in the fast path:
```
UNARY (non-raising)      -> id(dom0)
FORWARD_TYPE_RAISING     -> caps(result, result.l) @ id(dom0)      # result = cod.result
BACKWARD_TYPE_RAISING    -> id(dom0) @ caps(result.r, result)
FORWARD_APPLICATION      -> fa(left.result, right)
BACKWARD_APPLICATION     -> ba(left, right.result)
FORWARD_COMPOSITION      -> fc(left.left, left.right, right.right)
BACKWARD_COMPOSITION     -> bc(left.left, left.right, right.right)
FORWARD_CROSSED_COMP     -> fx(left.left, left.right, right.left)
BACKWARD_CROSSED_COMP    -> bx(left.right, left.left, right.right)
GEN_FORWARD_COMP   mid=left.argument:  id(left.result) @ cups(mid.l, mid) @ id(right[len(mid):])
GEN_BACKWARD_COMP  mid=right.argument: id(left[:-len(mid)]) @ cups(mid, mid.r) @ id(right.result)
GEN_FWD_CROSSED  mid=left.left; (l,join,r)=right.split(left.right):
    ( swap(mid<<join, l) @ id(join) >> id(l@mid) @ cups(join.l, join) ) @ id(r)
    # mid<<join == mid @ join.l
GEN_BWD_CROSSED  mid=right.right; (l,join,r)=left.split(right.left):
    id(l) @ ( id(join) @ swap(r, join>>mid) >> cups(join, join.r) @ id(mid@r) )
    # join>>mid == join.r @ mid
REMOVE_PUNCTUATION_LEFT  -> id(right)
REMOVE_PUNCTUATION_RIGHT -> id(left)
```
`_to_diagram` non-planar recursion (ccg_tree.py:458-530):
```
LEXICAL: punc -> (Id(), Id()); else (Word(text, cod).to_diagram(), Id(cod))
UNARY:   this_layer = swap(right, left)   # is_over: left=bt.left, right=bt.right.l
                                          # else:    left=bt.left.r, right=bt.right
binary:  child_types=[c.biclosed_type]; this_layer = rule.apply(child_types, biclosed_type)
combine: words, diag = [Id().tensor(*d) for d in zip(*children)]; diag >>= this_layer
return (words, diag)
final to_diagram: words >> diag
```

## File map

```
lambeq/backend/fast/build.py          fa/ba/fc/bc/fx/bx/gfc/gbc/gfx/gbx/ftr/btr/swaps (FTy->FDiagram)
lambeq/backend/fast/__init__.py       + re-export build
lambeq/text2diagram/ccg_tree.py       to_fast_diagram rewrite + _to_fast_diagram + _fast_rule_layer
tests/backend/test_fast_build.py      combinators vs grammar oracle
tests/backend/test_fast_construct.py  per-rule CCGTree + corpus differential
```

---

### Task 1: Builder combinators (`lambeq/backend/fast/build.py`)

**Files:** Create `lambeq/backend/fast/build.py`; modify
`lambeq/backend/fast/__init__.py`; Test `tests/backend/test_fast_build.py`.

- [ ] **Step 1: failing tests** (`tests/backend/test_fast_build.py`):

```python
import pytest

from lambeq.backend import grammar
from lambeq.backend.fast import build, convert
from lambeq.backend.fast.diagram import CUP, SWAP


def _g(ty):
    return convert.ty_to_fast(ty)


N, S, P = grammar.Ty('n'), grammar.Ty('s'), grammar.Ty('p')


@pytest.mark.parametrize('gfn,bfn,args', [
    (grammar.Diagram.fa, build.fa, (N, S)),
    (grammar.Diagram.fa, build.fa, (N @ S, P)),
    (grammar.Diagram.ba, build.ba, (N, S)),
    (grammar.Diagram.ba, build.ba, (N, S @ P)),
    (grammar.Diagram.fc, build.fc, (N, S, P)),
    (grammar.Diagram.bc, build.bc, (N, S, P)),
    (grammar.Diagram.fx, build.fx, (N, S, P)),
    (grammar.Diagram.bx, build.bx, (N, S, P)),
])
def test_combinator_matches_grammar(gfn, bfn, args):
    expected = gfn(*args)
    got = convert.to_grammar(bfn(*[_g(a) for a in args]))
    assert got == expected, (gfn.__name__, args)


def test_swaps_matches_grammar():
    for left, right in [(N, S), (N @ S, P), (N @ S, P @ N), (N, S @ P)]:
        expected = grammar.Diagram.swap(left, right)
        got = convert.to_grammar(build.swaps(_g(left), _g(right)))
        assert got == expected, (left, right)


def test_type_raising_matches_apply():
    # FTR: A -> T/(T\A); cod.result = T.  caps(T, T.l) @ id(A)
    T, A = S, N
    got = convert.to_grammar(build.ftr(_g(T), _g(A)))
    assert got == grammar.Diagram.caps(T, T.l) @ grammar.Id(A)
    got_b = convert.to_grammar(build.btr(_g(T), _g(A)))
    assert got_b == grammar.Id(A) @ grammar.Diagram.caps(T.r, T)


def test_gfc_gbc_match_grammar():
    # GFC: id(L) @ cups(M.l, M) @ id(tail);  GBC mirror
    L, M, tail = N, S, P     # right = M @ tail = S @ P, len(M)=1
    got = convert.to_grammar(
        build.gfc(_g(L), _g(M), _g(tail)))
    assert got == (grammar.Id(L) @ grammar.Diagram.cups(M.l, M)
                   @ grammar.Id(tail))
    pre = P
    got_b = convert.to_grammar(build.gbc(_g(pre), _g(M), _g(N)))
    assert got_b == (grammar.Id(pre) @ grammar.Diagram.cups(M, M.r)
                     @ grammar.Id(N))


def test_swaps_box_kinds():
    d = build.swaps(_g(N @ S), _g(P))
    assert all(b.kind == SWAP for b, _ in d.terms)
    assert d.dom == _g(N @ S @ P) and d.cod == _g(P @ N @ S)


def test_fa_is_cups():
    d = build.fa(_g(N), _g(S))
    assert all(b.kind == CUP for b, _ in d.terms)
```

NOTE: `build.gfx`/`build.gbx` are exercised through the corpus
differential gate in Task 2 (they need a `split` result; unit-testing
them in isolation requires reconstructing `CCGType.split`, which Task 2's
real trees provide). If a unit test here is wanted, build it from the
exact grammar expression in the oracle table; do not invent a shape.

- [ ] **Step 2: run, confirm failure** (`$PY -m pytest tests/backend/test_fast_build.py -q` -> ImportError).

- [ ] **Step 3: implement `lambeq/backend/fast/build.py`:**

```python
# (Apache header — copy the block from lambeq/backend/grammar.py)
"""Fast-core equivalents of the grammar CCG combinators.

Each function takes ``FTy`` arguments and returns an ``FDiagram``,
mirroring ``grammar.Diagram.{fa,ba,fc,bc,fx,bx}`` (grammar.py:769-792)
and the generalised/type-raising fragments in ``CCGRule.apply``
(ccg_rule.py:204-314). Pure fast-core: no text2diagram dependency.
"""

from __future__ import annotations

from lambeq.backend.fast.diagram import FDiagram, caps, cups, swap
from lambeq.backend.fast.types import FTy


def _id(ty: FTy) -> FDiagram:
    return FDiagram.id(ty)


def swaps(left: FTy, right: FTy) -> FDiagram:
    """Decomposed complex swap, mirroring grammar.Swap.__new__
    (grammar.py:1853): bubble each right atom left through the left
    block. Matches the oracle's expanded elementary-swap form."""
    nl = len(left)
    frontier = list(left.atoms + right.atoms)
    terms = []
    for start in range(len(right)):
        for i in range(nl - 1, -1, -1):
            off = start + i
            a, b = frontier[off], frontier[off + 1]
            terms.append((swap(FTy((a,)), FTy((b,))), off))
            frontier[off], frontier[off + 1] = b, a
    return FDiagram(left @ right, tuple(terms), right @ left)


def fa(left: FTy, right: FTy) -> FDiagram:
    return _id(left) @ cups(right.l, right)


def ba(left: FTy, right: FTy) -> FDiagram:
    return cups(left, left.r) @ _id(right)


def fc(left: FTy, middle: FTy, right: FTy) -> FDiagram:
    return _id(left) @ cups(middle.l, middle) @ _id(right.l)


def bc(left: FTy, middle: FTy, right: FTy) -> FDiagram:
    return _id(left.r) @ cups(middle, middle.r) @ _id(right)


def fx(left: FTy, middle: FTy, right: FTy) -> FDiagram:
    return (_id(left) @ swaps(middle.l, right.r) @ _id(middle)
            >> swaps(left, right.r) @ cups(middle.l, middle))


def bx(left: FTy, middle: FTy, right: FTy) -> FDiagram:
    return (_id(middle) @ swaps(left.l, middle.r) @ _id(right)
            >> cups(middle, middle.r) @ swaps(left.l, right))


def gfc(left: FTy, middle: FTy, tail: FTy) -> FDiagram:
    # GEN_FORWARD_COMP: id(left.result) @ cups(mid.l, mid) @ id(right[len(mid):])
    return _id(left) @ cups(middle.l, middle) @ _id(tail)


def gbc(prefix: FTy, middle: FTy, right: FTy) -> FDiagram:
    # GEN_BACKWARD_COMP: id(left[:-len(mid)]) @ cups(mid, mid.r) @ id(right.result)
    return _id(prefix) @ cups(middle, middle.r) @ _id(right)


def gfx(mid: FTy, l: FTy, join: FTy, r: FTy) -> FDiagram:
    # ( swap(mid<<join, l) @ id(join) >> id(l@mid) @ cups(join.l, join) ) @ id(r)
    inner = (swaps(mid @ join.l, l) @ _id(join)
             >> _id(l @ mid) @ cups(join.l, join))
    return inner @ _id(r)


def gbx(mid: FTy, l: FTy, join: FTy, r: FTy) -> FDiagram:
    # id(l) @ ( id(join) @ swap(r, join>>mid) >> cups(join, join.r) @ id(mid@r) )
    inner = (_id(join) @ swaps(r, join.r @ mid)
             >> cups(join, join.r) @ _id(mid @ r))
    return _id(l) @ inner


def ftr(result: FTy, dom0: FTy) -> FDiagram:
    return caps(result, result.l) @ _id(dom0)


def btr(result: FTy, dom0: FTy) -> FDiagram:
    return _id(dom0) @ caps(result.r, result)
```

Re-export from `lambeq/backend/fast/__init__.py`: add
`from lambeq.backend.fast import build` and include `'build'` in
`__all__` (import the module, not each function).

- [ ] **Step 4: green** (`$PY -m pytest tests/backend/test_fast_build.py -q`), **Step 5: flake8 + commit**

```bash
$PY -m flake8 lambeq/backend/fast/build.py
git add lambeq/backend/fast tests/backend/test_fast_build.py
git commit -m "Add fast-core CCG combinator builders

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

If a combinator test fails: the oracle (`grammar.Diagram.X` / the
`CCGRule.apply` expression) is the truth — match it, do not edit the
test's expected side.

---

### Task 2: Direct recursion (`ccg_tree.py`) + differential gates

**Files:** Modify `lambeq/text2diagram/ccg_tree.py`; Test
`tests/backend/test_fast_construct.py`.

- [ ] **Step 1: failing tests** (`tests/backend/test_fast_construct.py`):

```python
from lambeq.backend.fast import convert


def test_corpus_construction_matches_oracle(bobcat_trees):
    assert bobcat_trees
    for t in bobcat_trees:
        assert (convert.to_grammar(t.to_fast_diagram())
                == t.to_diagram()), t.text if hasattr(t, 'text') else t


def test_construction_no_grammar_diagram(bobcat_trees):
    # to_fast_diagram returns an FDiagram, not a grammar.Diagram
    from lambeq.backend.fast import FDiagram
    for t in bobcat_trees[:5]:
        assert isinstance(t.to_fast_diagram(), FDiagram)
```

(The existing `tests/backend/test_fast_convert.py::test_to_fast_diagram_equivalent`
also gates this and now becomes a real differential test — keep it.)

- [ ] **Step 2: run** — `test_to_fast_diagram_equivalent` currently
passes trivially (wrapper). Confirm the new tests pass with the wrapper
too (they should — wrapper already returns an FDiagram equal to the
oracle). This task REPLACES the wrapper; the gate must STAY green after
the replacement. So: implement Step 3, then these are the regression
gate.

- [ ] **Step 3: implement** in `lambeq/text2diagram/ccg_tree.py`.

Replace `to_fast_diagram` (currently the `convert.to_fast(self.to_diagram())`
wrapper at ccg_tree.py:449) with:

```python
    def to_fast_diagram(self):
        """Build a fast-core ``FDiagram`` directly from the resolved
        CCG tree, without allocating a ``grammar.Diagram``."""
        resolved = self.collapse_noun_phrases()._resolved()
        words, grammar = resolved._to_fast_diagram()
        return words >> grammar

    def _to_fast_diagram(self):
        from lambeq.backend.fast import build
        from lambeq.backend.fast.diagram import FDiagram, word
        from lambeq.backend.fast.convert import ty_to_fast
        from lambeq.text2diagram.ccg_rule import CCGRule
        from lambeq.text2diagram.ccg_type import CCGType

        if self.rule == CCGRule.LEXICAL:
            if self.biclosed_type == CCGType.PUNCTUATION:
                return FDiagram.id(), FDiagram.id()
            cod = ty_to_fast(self.biclosed_type.to_grammar())
            return word(self.text, cod).to_diagram(), FDiagram.id(cod)

        if self.rule == CCGRule.UNARY:
            if self.biclosed_type.is_over:
                left = ty_to_fast(self.biclosed_type.left.to_grammar())
                right = ty_to_fast(self.biclosed_type.right.to_grammar()).l
            else:
                left = ty_to_fast(self.biclosed_type.left.to_grammar()).r
                right = ty_to_fast(self.biclosed_type.right.to_grammar())
            this_layer = build.swaps(right, left)
        else:
            this_layer = _fast_rule_layer(
                self.rule, [c.biclosed_type for c in self.children],
                self.biclosed_type)

        children = [child._to_fast_diagram() for child in self.children]
        words_parts, diag_parts = zip(*children)
        words = FDiagram.id()
        for w in words_parts:
            words = words @ w
        diag = FDiagram.id()
        for d in diag_parts:
            diag = diag @ d
        diag = diag >> this_layer
        return words, diag
```

Add the module-level dispatch `_fast_rule_layer` (transcribe
`CCGRule.apply`, ccg_rule.py:204-314) near the bottom of `ccg_tree.py`:

```python
def _fast_rule_layer(rule, dom, cod):
    """Fast-core analogue of CCGRule.apply for resolved binary/unary
    type-raising rules. Mirrors ccg_rule.py:204-314."""
    from lambeq.backend.fast import build
    from lambeq.backend.fast.diagram import FDiagram
    from lambeq.backend.fast.convert import ty_to_fast
    from lambeq.text2diagram.ccg_rule import CCGRule

    def f(ccgtype):
        return ty_to_fast(ccgtype.to_grammar())

    if rule in (CCGRule.BACKWARD_TYPE_RAISING,
                CCGRule.FORWARD_TYPE_RAISING):
        result = f(cod.result)
        dom0 = f(dom[0])
        if rule == CCGRule.BACKWARD_TYPE_RAISING:
            return build.btr(result, dom0)
        return build.ftr(result, dom0)

    left, right = dom
    if rule == CCGRule.FORWARD_APPLICATION:
        return build.fa(f(left.result), f(right))
    if rule == CCGRule.BACKWARD_APPLICATION:
        return build.ba(f(left), f(right.result))
    if rule == CCGRule.FORWARD_COMPOSITION:
        return build.fc(f(left.left), f(left.right), f(right.right))
    if rule == CCGRule.BACKWARD_COMPOSITION:
        return build.bc(f(left.left), f(left.right), f(right.right))
    if rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
        return build.fx(f(left.left), f(left.right), f(right.left))
    if rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
        return build.bx(f(left.right), f(left.left), f(right.right))
    if rule == CCGRule.GENERALIZED_FORWARD_COMPOSITION:
        mid = f(left.argument)
        return build.gfc(f(left.result), mid, f(right)[len(mid):])
    if rule == CCGRule.GENERALIZED_BACKWARD_COMPOSITION:
        mid = f(right.argument)
        return build.gbc(f(left)[:len(f(left)) - len(mid)], mid,
                         f(right.result))
    if rule == CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION:
        mid = f(left.left)
        gl, join, gr = right.split(left.right)
        return build.gfx(mid, f(gl), f(join), f(gr))
    if rule == CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION:
        mid = f(right.right)
        gl, join, gr = left.split(right.left)
        return build.gbx(mid, f(gl), f(join), f(gr))
    if rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
        return FDiagram.id(f(right))
    if rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
        return FDiagram.id(f(left))
    raise AssertionError(f'unreachable rule {rule}')
```

IMPORTANT details:
- `CCGType.split` returns 3 `CCGType`s; convert each with `f`. Verify the
  signature against `ccg_type.py` (`right.split(left.right)`); if `split`
  returns something else, match the real API — the corpus gate is the
  arbiter.
- `_resolved()` may collapse trivial UNARY nodes, so the UNARY branch
  only fires for direction-change swaps (matches `_to_diagram`).
- The `words/diag` accumulation uses left-fold `@` to mirror
  `Id().tensor(*parts)`. For a single child the fold yields that child's
  diagram unchanged (tensor with `Id()` is identity).
- Do not import `build`/`convert` at module top if it risks an import
  cycle; the function-local imports above avoid it (mirrors the existing
  `to_fast_diagram` which imports convert locally).

- [ ] **Step 4: green** — `$PY -m pytest tests/backend/test_fast_construct.py tests/backend/test_fast_convert.py -q` (600000ms). The corpus differential (`test_corpus_construction_matches_oracle` + the existing `test_to_fast_diagram_equivalent`) is THE gate. Debug a mismatch by printing the first failing tree's `to_diagram()` vs `to_grammar(to_fast_diagram())` layer reprs and finding the diverging rule.

- [ ] **Step 5: flake8 + commit**

```bash
$PY -m flake8 lambeq/text2diagram/ccg_tree.py
git add lambeq/text2diagram/ccg_tree.py tests/backend/test_fast_construct.py
git commit -m "Build FDiagram directly from the resolved CCG tree

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Full-corpus gate + benchmark + RESULTS + push

**Files:** Modify `benchmarks/RESULTS.md` (and read
`benchmarks/fastdiag_bench.py` — its construction row already times
`to_diagram` vs `to_fast_diagram`).

- [ ] **Step 1: full-corpus differential.** Run a one-off check (throwaway,
not committed) over the full corpus to confirm the direct path equals the
oracle for EVERY parseable diagram:

```bash
$PY - <<'PY'
from lambeq import BobcatParser, VerbosityLevel
from lambeq.backend.fast import convert
p = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value, parser_backend='rust')
sents = [l.split() for l in open('/tmp/coco_bench.txt') if l.strip()][:2000]
trees = [t for t in p.sentences2trees(sents, tokenised=True, suppress_exceptions=True) if t]
bad = 0
for t in trees:
    if convert.to_grammar(t.to_fast_diagram()) != t.to_diagram():
        bad += 1
print(f'{len(trees)} trees, {bad} mismatches')
PY
```
Expected: `0 mismatches`. If not, STOP — report the failing rule (do not
proceed to RESULTS); the gate must be clean.

- [ ] **Step 2: benchmark.** Run `$PY benchmarks/fastdiag_bench.py
/tmp/coco_bench.txt --num 2000` (or the largest that finishes; note the
count). Capture the construction row (old `to_diagram` vs fast
`to_fast_diagram`). Confirm the script's round-trip gate still prints
`N/N identical`.

- [ ] **Step 3: update RESULTS.** In `benchmarks/RESULTS.md`, find the
existing "Fast diagram core" construction line/row and update it to the
new direct-recursion number, OR append a short subsection
`### Direct construction` with: the before (wrapper ~2.4ms) vs after
(direct) ms/diagram and speedup, the resolved/assembly split (0.14ms /
2.06ms) that motivated it, and whether the <=0.4ms target was met. Honest
one-line profile if missed. Paste real numbers from Step 2.

- [ ] **Step 4: full sweep + commit + push**

```bash
$PY -m pytest tests/backend/test_fast_build.py tests/backend/test_fast_construct.py tests/backend/test_fast_convert.py tests/backend/test_fast_types.py tests/backend/test_fast_diagram.py -q
git add benchmarks/RESULTS.md
git commit -m "Record direct-construction benchmark results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin fast-construction
```

- [ ] **Step 5: report** — the new construction ms/diagram + speedup,
the full-corpus differential result (N/N), whether the target was met,
and whether push succeeded.
