# Fast SpiderAnsatz (FFunctor-based) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `FSpiderAnsatz`, a pure-Python `FFunctor`-based ansatz mapping `FDiagram → FDiagram` (Dim-typed wires + `Symbol` payloads), numerically identical to the legacy `SpiderAnsatz`, capturing the ~2.06ms object-churn cost.

**Architecture:** One `FFunctor(ob, ar)` pass. `ob` maps each pregroup atom to its Dim atoms (via `register_dim`). `ar` ports `SpiderAnsatz._split_ar` (high-arity → spider chains) and `TensorAnsatz._ar` (Symbol creation) onto `FBox`/`FTy`, returning the split fragment as an `FDiagram` with `Symbol` payloads. Symbol names reuse the legacy `_summarise_box` (via a cheap reconstructed `grammar.Box`); directed dims port `_generate_directed_dom_cod`. The legacy `SpiderAnsatz` is the untouched oracle.

**Tech Stack:** Pure Python. Fast core (`lambeq/backend/fast/`: `FFunctor`, `FBox`, `FDiagram`, `convert.{ty_to_grammar,ty_to_fast,register_dim,to_fast}`, `contraction`), `lambeq/ansatz/{tensor,base}.py` (oracle + reused `_summarise_box`), `lambeq/backend/symbol.Symbol`, `PytorchModel`.

**Spec:** `docs/superpowers/specs/2026-06-14-fast-ansatz-design.md`

---

## Conventions (every task)

- Repo `/home/kinianlo/projects/lambeq`, branch `fast-ansatz`.
  `PY=~/.pyenv/versions/qnlp/bin/python`. 600000ms timeouts on Bobcat-
  loading commands. Corpus `/tmp/coco_bench.txt`.
- Commit style: plain sentence + trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- ADDITIVE only: new `lambeq/backend/fast/ansatz.py` + re-export + tests +
  benchmark. The legacy `lambeq/ansatz/` and the fast core are NOT
  modified.

## Oracle facts (verified, do not re-derive)

- `SpiderAnsatz._split_ar` (tensor.py:255-274): if `len(dom)+len(cod) <=
  max_order` → `Box(f'{name}_0', dom, cod, z=ar.z)` (rename only). Elif
  non-empty dom (`uncurry.matches`) → `split_functor(uncurry.rewrite(ar))`
  where uncurry-left rewrite (rewrite/base.py) =
  `Id(dom) @ Box(name, (), dom.r @ cod) >> Cup(dom, dom.r) @ Id(cod)`.
  Else (wide cod) → spider chain: `step=max_order-1`; for each cod slice
  `cod[start:start+step+1]` a `Word(f'{name}_{i}', slice)`, joined by
  `Spider(slice[-1:], 2, 1)`; assembled `Id().tensor(*boxes) >>
  Id().tensor(*spiders)`.
- `TensorAnsatz._ar` (tensor.py:54-66): `Symbol(_summarise_box(box),
  directed_dom=dd.product, directed_cod=dc.product)`; box dom/cod mapped
  by the ob functor.
- `_summarise_box` (base.py:62-71): `f'{name}_{dom}_{cod}'` with `dom =
  str(box.dom).replace(' @ ','@') if box.dom else ''` (same for cod), then
  `.translate({ord(c): f'\\{c}' for c in ':, '})`.
- `_generate_directed_dom_cod` (tensor.py:68-109): flow-dom/flow-cod by
  z-parity — for each atom in `box.cod`: `z%2` → flow_dom else flow_cod;
  for each atom in `box.dom`: `z%2` → flow_cod else flow_dom. Returns
  `(functor(flow_dom), functor(flow_cod))` (Dims); `.product` = product of
  the atoms' dims (empty → 1).
- Adjoint atoms map to the SAME Dim as the base (`Dim.rotate`=identity).
- `convert.register_dim(n)` interns `#<int>` and records `dim_of`;
  `Dim(a,b).dim == (a, b)`. `convert.ty_to_grammar(fty)`,
  `convert.ty_to_fast(ty)`. `atom_name(id)`, `atom_z(id)` from
  `fast.types`.
- FFunctor: `FFunctor(ob, ar)`; `ob(functor, atom_id) -> FTy`;
  `ar(functor, fbox) -> FBox | FDiagram`; cache-by-FBox; structural boxes
  (CUP/CAP/SWAP/SPIDER) are auto-mapped by FFunctor using `ob` UNLESS we
  want custom handling — VERIFY: read `functor.py` to confirm whether
  structural boxes bypass `ar`. (They do in the oracle. For the ansatz,
  structural boxes need their TYPES dim-mapped but no symbol — FFunctor's
  built-in structural mapping already does exactly this via `ob`. So `ar`
  only handles PLAIN/WORD.)
- `PytorchModel.from_diagrams(circuits)` → `.symbols` (sorted Symbol
  list), `.initialise_weights()` → `.weights` (ParameterList,
  `torch.rand(sym.size)`); `dict(zip(symbols, weights))` keyed by Symbol
  (hashed by name). `contraction.evaluate(spec, weights_by_symbol)`.

## File map

```
lambeq/backend/fast/ansatz.py     FSpiderAnsatz
lambeq/backend/fast/__init__.py   + re-export FSpiderAnsatz
tests/backend/test_fast_ansatz.py structural + symbol-parity + numeric gate
benchmarks/fastansatz_bench.py    legacy vs fast ansatz throughput
benchmarks/RESULTS.md             + section
```

---

### Task 1: `FSpiderAnsatz` (`lambeq/backend/fast/ansatz.py`)

**Files:** Create `lambeq/backend/fast/ansatz.py`; modify
`lambeq/backend/fast/__init__.py`; Test `tests/backend/test_fast_ansatz.py`.

FIRST: read `lambeq/backend/fast/functor.py` to confirm (a) the `ob`/`ar`
callable signatures, (b) that structural boxes (CUP/CAP/SWAP/SPIDER) are
auto-mapped via `ob` and do NOT hit `ar` (so `ar` only handles
PLAIN/WORD), and (c) how `ar` returning an `FDiagram` is spliced. Adjust
the code below to the real signatures if they differ.

- [ ] **Step 1: implement `lambeq/backend/fast/ansatz.py`:**

```python
# (Apache header — copy from lambeq/backend/grammar.py)
"""FFunctor-based SpiderAnsatz on the fast diagram core."""

from __future__ import annotations

from collections.abc import Mapping

from lambeq.ansatz.base import BaseAnsatz
from lambeq.backend import grammar
from lambeq.backend.fast import convert
from lambeq.backend.fast.diagram import (CAP, CUP, FBox, FDiagram, SPIDER,
                                         SWAP, WORD, spider, word)
from lambeq.backend.fast.functor import FFunctor
from lambeq.backend.fast.types import atom_name, atom_z, FTy
from lambeq.backend.symbol import Symbol
from lambeq.backend.tensor import Dim


class FSpiderAnsatz:
    """SpiderAnsatz on the fast core. Numerically identical to
    lambeq.ansatz.SpiderAnsatz; ~10x faster (no grammar.Diagram churn)."""

    def __init__(self, ob_map: Mapping[grammar.Ty, Dim],
                 max_order: int = 2) -> None:
        if max_order < 2:
            raise ValueError('`max_order` must be at least 2')
        self.ob_map = dict(ob_map)
        self.max_order = max_order
        # integer dim per base atomic type name
        self._dim_of_name = {ty.name: int(dim.product)
                             for ty, dim in self.ob_map.items()}
        self.functor = FFunctor(ob=self._ob, ar=self._ar)

    # --- ob: pregroup atom -> Dim atoms (adjoint maps to base dim) ---
    def _ob(self, functor, atom_id: int) -> FTy:
        name = atom_name(atom_id)
        dim = self.ob_map[grammar.Ty(name)]          # base (z=0) lookup
        return FTy(tuple(convert.register_dim(d) for d in dim.dim))

    def __call__(self, diagram: FDiagram) -> FDiagram:
        return self.functor(diagram)

    # --- ar: split + symbolise one PLAIN/WORD box ---
    def _ar(self, functor, box: FBox):
        n = len(box.dom) + len(box.cod)
        if n <= self.max_order:
            return self._sym_box(f'{box.name}_0', box.dom, box.cod,
                                 box.z, box.kind)
        if len(box.dom):
            # uncurry-left: Id(dom) @ Box(name,(),dom.r@cod)
            #               >> Cup(dom, dom.r) @ Id(cod)
            return self._uncurry(box)
        # wide cod: spider chain
        return self._spider_chain(box)

    # build one symbol-carrying fast box, dims mapped via _ob
    def _sym_box(self, name, fdom, fcod, z, kind):
        sym = self._symbol(name, fdom, fcod, z)
        mdom = self._map_ty(fdom)
        mcod = self._map_ty(fcod)
        return FBox(name, mdom, mcod, kind, z, False, sym).to_diagram()

    def _map_ty(self, fty: FTy) -> FTy:
        out: tuple[int, ...] = ()
        for a in fty.atoms:
            out = out + self._ob(self.functor, a).atoms
        return FTy(out)

    # Symbol name + directed dims, byte-identical to the legacy ansatz.
    def _symbol(self, name, fdom, fcod, z) -> Symbol:
        gbox = grammar.Box(name, convert.ty_to_grammar(fdom),
                           convert.ty_to_grammar(fcod), z=z)
        sym_name = BaseAnsatz._summarise_box(gbox)
        dd, dc = self._directed_products(gbox)
        return Symbol(sym_name, directed_dom=dd, directed_cod=dc)

    def _directed_products(self, gbox) -> tuple[int, int]:
        # port of _generate_directed_dom_cod -> (dd.product, dc.product)
        dd = dc = 1
        for ty in gbox.cod:
            d = self._dim_of_name[ty.name]
            if ty.z % 2:
                dd *= d
            else:
                dc *= d
        for ty in gbox.dom:
            d = self._dim_of_name[ty.name]
            if ty.z % 2:
                dc *= d
            else:
                dd *= d
        return dd, dc

    def _uncurry(self, box: FBox) -> FDiagram:
        dom, cod = box.dom, box.cod
        # new wide-cod box, then recurse split on it
        new_cod = dom.r @ cod
        inner = self._ar(self.functor,
                         FBox(box.name, FTy(), new_cod, WORD, box.z))
        # Id(dom) @ inner  >>  cups(dom, dom.r) @ Id(cod), all dim-mapped
        from lambeq.backend.fast.diagram import cups
        mdom, mcod = self._map_ty(dom), self._map_ty(cod)
        left = FDiagram.id(mdom) @ inner
        right = cups(mdom, self._map_ty(dom.r)) @ FDiagram.id(mcod)
        return left >> right

    def _spider_chain(self, box: FBox) -> FDiagram:
        step = self.max_order - 1
        cod = box.cod
        boxes = []
        for i, start in enumerate(range(0, len(cod) - 1, step)):
            sl = cod[start:start + step + 1]            # FTy slice
            boxes.append((f'{box.name}_{i}', sl))
        # words tensored, then spiders joining adjacent slices' last atom
        words = FDiagram.id(FTy())
        for nm, sl in boxes:
            words = words @ self._sym_box(nm, FTy(), sl, box.z, WORD)
        # spiders: one Spider(slice[-1:],2,1) between consecutive slices
        result = words
        # See _split_ar (tensor.py:264-274): build the spider layer and
        # compose. Port the exact offset structure; the numeric gate is
        # the check. Spiders use the MAPPED type of slice[-1:].
        result = self._spider_layer(result, boxes, box.z)
        return result
```

NOTE on `_spider_chain`/`_spider_layer` (the fiddly part). Read
`SpiderAnsatz._split_ar` tensor.py:255-274 in full. The wide-cod branch
returns `Id().tensor(*boxes) >> Id().tensor(*spiders)` where
`boxes[i] = Word(f'{name}_{i}', cod[start:start+step+1])` and
`spiders = [Id(cod[:1])] + per-slice [Id(slice[1:-1]),
Spider(slice[-1:],2,1)]` with the last entry replaced by
`Id(spiders[-1].cod)`. TWO acceptable implementations (the Task-2 numeric
gate over real COCO wide-cod words decides correctness either way; do NOT
weaken the gate):
  (a) **Port** it onto fast constructors (`FDiagram.id`, `word`,
      `spider(ty,2,1)`, `@`, `>>`), dim-mapping every type, mirroring the
      oracle's offsets exactly. Highest-throughput.
  (b) **Hybrid fallback** (lower risk, split boxes are rare): for the
      wide-cod case only, call the legacy
      `SpiderAnsatz(self.ob_map, self.max_order)._split_ar(None, gbox)`
      to get the grammar split fragment, then run the legacy second
      functor on it (`SpiderAnsatz(...).functor(fragment)`) to get the
      tensorised fragment, and `convert.to_fast(...)` it (which carries
      the `Symbol`s as payloads). Splice that FDiagram. The common
      `<= max_order` path stays the fast `_sym_box` route, so the bulk of
      the win is preserved.
Prefer (a); if the offsets prove troublesome, ship (b) and note it in
RESULTS. Either way the numeric gate must pass over the full corpus.

Re-export `FSpiderAnsatz` from `lambeq/backend/fast/__init__.py`.

- [ ] **Step 2: structural + symbol-parity unit tests**
(`tests/backend/test_fast_ansatz.py`):

```python
import pytest

from lambeq import AtomicType, SpiderAnsatz
from lambeq.backend import grammar
from lambeq.backend.fast import FSpiderAnsatz, convert
from lambeq.backend.fast.diagram import WORD
from lambeq.backend.tensor import Dim

N, S = AtomicType.NOUN, AtomicType.SENTENCE
OB = {N: Dim(2), S: Dim(3)}


def _symbols(fd):
    return {(b.payload.name, b.payload.size)
            for b, _ in fd.terms if b.payload is not None}


def test_simple_word_symbol_parity():
    # a within-max_order word: cod n@s (2 legs)
    g = grammar.Word('w', grammar.Ty('n') @ grammar.Ty('s'))
    legacy = SpiderAnsatz(OB)(g.to_diagram())
    fast = FSpiderAnsatz(OB)(convert.to_fast(g.to_diagram()))
    legacy_syms = {(s.name, s.size) for s in legacy.free_symbols}
    assert _symbols(fast) == legacy_syms


def test_wide_cod_splits():
    # cod width 3 (> max_order 2) -> spider chain
    n, s = grammar.Ty('n'), grammar.Ty('s')
    g = grammar.Word('likes', n @ s @ n)
    legacy = SpiderAnsatz(OB)(g.to_diagram())
    fast = FSpiderAnsatz(OB)(convert.to_fast(g.to_diagram()))
    assert _symbols(fast) == {(s.name, s.size)
                              for s in legacy.free_symbols}
    # the split introduced spider boxes
    from lambeq.backend.fast.diagram import SPIDER
    assert any(b.kind == SPIDER for b, _ in fast.terms)


def test_ob_adjoint_same_dim():
    n = grammar.Ty('n')
    fd_dom = convert.to_fast((n.r).to_diagram() if hasattr(n.r, 'to_diagram')
                             else grammar.Id(n.r))
    # adjoint n.r maps to the same Dim(2) as n
    ans = FSpiderAnsatz(OB)
    from lambeq.backend.fast.types import atom
    assert ans._ob(None, atom('n', 1)) == ans._ob(None, atom('n', 0))
```

(Adjust the adjoint test mechanics to the real `_ob` signature; the
substance is: `n.r`/`n.l` atoms map to the same Dim as `n`.)

- [ ] **Step 3: run** `$PY -m pytest tests/backend/test_fast_ansatz.py -q`
(the simple/wide/ob tests; no Bobcat model needed). Green.
- [ ] **Step 4: flake8 + commit**

```bash
$PY -m flake8 lambeq/backend/fast/ansatz.py
git add lambeq/backend/fast tests/backend/test_fast_ansatz.py
git commit -m "Add FFunctor-based fast SpiderAnsatz

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

If `_symbols` parity fails: the symbol name or directed dims differ from
the legacy ansatz — diff `_summarise_box`/`_directed_products` output
against the legacy box; the legacy is the truth. If the wide-cod split
shape is wrong, fix `_spider_chain`/`_spider_layer` against
`_split_ar`. Do NOT weaken parity.

---

### Task 2: Corpus numeric-equivalence gate

**Files:** add to `tests/backend/test_fast_ansatz.py`.

- [ ] **Step 1: the gate** (mirrors `test_fast_contraction`):

```python
def test_numeric_pipeline_matches_pytorch_model(bobcat_diagrams):
    torch = pytest.importorskip('torch')
    from lambeq import PytorchModel, RemoveCupsRewriter
    from lambeq.backend.fast import contraction
    from lambeq.backend.fast.ansatz import FSpiderAnsatz

    ob = {t: Dim(2) for t in AtomicType}
    rc = RemoveCupsRewriter()
    g_circuits = [SpiderAnsatz(ob)(rc(d)) for d in bobcat_diagrams[:10]]

    model = PytorchModel.from_diagrams(g_circuits)
    torch.manual_seed(0)
    model.initialise_weights()
    expected = model.get_diagram_output(g_circuits)
    weights = dict(zip(model.symbols, model.weights))

    fans = FSpiderAnsatz(ob)
    for d, exp in zip(bobcat_diagrams[:10], expected):
        fd = fans(convert.to_fast(rc(d)))
        spec = contraction.to_contraction(fd)
        got = contraction.evaluate(spec, weights)
        assert torch.allclose(got, exp, atol=1e-5), d
```

This is THE gate: the fast ansatz + contraction must produce tensors
identical to the legacy ansatz + PytorchModel, weights matched by Symbol
name. Splitting (wide-cod words) is exercised by real COCO words.

- [ ] **Step 2: run** `$PY -m pytest tests/backend/test_fast_ansatz.py -q`
(600000ms; loads Bobcat). All pass. Debug a mismatch by shrinking to the
first failing diagram, comparing the fast circuit's symbols/structure to
the legacy `tensor.Diagram`; the bug is in `_ar`/split or `_ob` — fix
against the oracle, never weaken `atol`.

- [ ] **Step 3: commit**

```bash
git add tests/backend/test_fast_ansatz.py
git commit -m "Gate the fast ansatz numerically against PytorchModel

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Benchmark + RESULTS + push

**Files:** Create `benchmarks/fastansatz_bench.py`; modify
`benchmarks/RESULTS.md`.

- [ ] **Step 1: write `benchmarks/fastansatz_bench.py`** — parse the
corpus (rust backend), `remove_cups`, then time over all diagrams:
- legacy: `[SpiderAnsatz(ob)(d) for d in nocups]`
- fast: pre-convert `fasts=[convert.to_fast(d) for d in nocups]`, time
  `[FSpiderAnsatz(ob)(fd) for fd in fasts]` (the ansatz proper); also
  report the convert cost separately.
Report ms/diagram legacy vs fast + speedup. Best-of-3, warm-up. argparse
+ `--num`. Before timing, assert numeric equivalence on a few diagrams
(reuse the Task-2 harness) so numbers come from a correct build.
Structure like `benchmarks/fastdiag_bench.py`.

- [ ] **Step 2: run** `$PY benchmarks/fastansatz_bench.py
/tmp/coco_bench.txt --num 1000` (or largest that finishes; note count).
Capture legacy vs fast ms/diagram + speedup.

- [ ] **Step 3: append `## Fast SpiderAnsatz` to `benchmarks/RESULTS.md`**
(match style): legacy 2.06ms baseline vs fast ms/diagram + speedup,
whether the ~0.1-0.2ms target was met, and an honest one-line profile of
the residual (e.g. Symbol object construction / per-box grammar.Box
reconstruction for naming is the irreducible Python cost). Note this is a
once-per-dataset preprocessing win, not a training-step win. PASTE REAL
NUMBERS.

- [ ] **Step 4: full sweep + commit + push**

```bash
$PY -m pytest tests/backend/test_fast_ansatz.py tests/backend/test_fast_contraction.py -q
$PY -m flake8 benchmarks/fastansatz_bench.py
git add benchmarks/fastansatz_bench.py benchmarks/RESULTS.md
git commit -m "Benchmark the fast SpiderAnsatz against the legacy ansatz

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin fast-ansatz
```

- [ ] **Step 5: report** — legacy vs fast ms/diagram + speedup, whether
the target was met, numeric gate result, whether push succeeded.
