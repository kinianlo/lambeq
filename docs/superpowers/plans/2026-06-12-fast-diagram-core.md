# Fast Diagram Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new immutable, integer-based diagram core (`lambeq.backend.fast`) that is >=10x faster than `backend.grammar` on construction, rewriting, functor application and copying, interoperating via lossless shims.

**Architecture:** Interned atom ids + tuple types (`FTy`), frozen slotted boxes with stored hashes (`FBox`), diagrams as `(box, offset)` term tuples with stored cod/hash (`FDiagram`). Validation behind a switch. Copy-free functors (immutability makes cache hits shared references). Cup/snake removal ported onto the offset arrays. A one-off `to_contraction()` extraction replaces per-step deepcopy+mutate in training.

**Tech Stack:** Pure Python (3.10+), pytest; oracles are the in-repo `lambeq/backend/grammar.py`, `snake_removal.py`, and `PytorchModel`.

**Spec:** `docs/superpowers/specs/2026-06-12-fast-diagram-core-design.md`

---

## Conventions (every task)

- Repo `/home/kinianlo/projects/lambeq`, branch `fast-diagram-core`.
  `PY=~/.pyenv/versions/qnlp/bin/python`. 600000ms timeouts on anything
  loading the Bobcat model.
- Oracles win over prompt summaries: `lambeq/backend/grammar.py`
  (Ty/Box/Layer/Diagram/Functor), `lambeq/backend/snake_removal.py`,
  `lambeq/backend/tensor.py` (Dim), `lambeq/text2diagram/ccg_tree.py`
  (`to_diagram`/`_to_diagram`).
- New tests live in `tests/backend/test_fast_*.py` and ALWAYS run with
  validation ON (a conftest fixture enables it).
- Corpus fixtures: `/tmp/coco_bench.txt` (2000 captions; regenerate via
  the snippet in Task 3 if missing).
- Commit style: plain sentence + trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File map

```
lambeq/backend/fast/__init__.py    public surface re-exports
lambeq/backend/fast/types.py       atom intern table + FTy
lambeq/backend/fast/diagram.py     FBox, FDiagram, primitive constructors
lambeq/backend/fast/validate.py    validation switch + checks
lambeq/backend/fast/convert.py     to_fast / to_grammar / Dim handling
lambeq/backend/fast/functor.py     FFunctor
lambeq/backend/fast/normal.py      cup/snake removal, interchange, normal form
lambeq/backend/fast/contraction.py ContractionSpec + torch evaluator
lambeq/backend/fast/model.py       FastPytorchModel (spec-cached drop-in)
lambeq/text2diagram/ccg_tree.py    + to_fast_diagram()
benchmarks/fastdiag_bench.py       the five-op benchmark
tests/backend/test_fast_types.py
tests/backend/test_fast_diagram.py
tests/backend/test_fast_convert.py
tests/backend/test_fast_functor.py
tests/backend/test_fast_normal.py
tests/backend/test_fast_contraction.py
```

---

### Task 1: Types (`types.py`) + validation switch (`validate.py`)

**Files:** Create `lambeq/backend/fast/__init__.py`,
`lambeq/backend/fast/types.py`, `lambeq/backend/fast/validate.py`,
`tests/backend/test_fast_types.py`, `tests/backend/__init__.py` (empty),
`tests/backend/conftest.py`.

- [ ] **Step 1: failing tests** — `tests/backend/conftest.py`:

```python
import pytest

from lambeq.backend import fast


@pytest.fixture(autouse=True)
def _validation_on():
    fast.set_validation(True)
    yield
    fast.set_validation(False)
```

`tests/backend/test_fast_types.py`:

```python
from lambeq.backend import fast
from lambeq.backend.fast.types import FTy, atom, atom_name, atom_z


def test_atom_interning():
    a = atom('n')
    assert atom('n') == a            # same id for same (name, z)
    assert atom('n', z=1) != a
    assert atom_name(a) == 'n' and atom_z(a) == 0


def test_fty_algebra():
    n, s = FTy.of('n'), FTy.of('s')
    ns = n @ s
    assert len(ns) == 2
    assert ns == FTy.of('n') @ FTy.of('s')
    assert hash(ns) == hash(FTy.of('n') @ FTy.of('s'))
    assert ns @ FTy() == ns          # empty type is the unit
    assert list(iter(ns)) == [n.atoms[0], s.atoms[0]]
    assert ns[0:1] == n


def test_adjoints_round_trip():
    n = FTy.of('n')
    assert n.l.r == n and n.r.l == n
    ns = FTy.of('n') @ FTy.of('s')
    assert ns.l.atoms == (FTy.of('s').l.atoms[0], FTy.of('n').l.atoms[0])
    assert ns.l.r == ns
    assert atom_z(n.l.atoms[0]) == atom_z(n.atoms[0]) - 1  # convention pinned


def test_validation_switch():
    assert fast.validation_enabled()
    fast.set_validation(False)
    assert not fast.validation_enabled()
    fast.set_validation(True)
    with fast.no_validation():
        assert not fast.validation_enabled()
    assert fast.validation_enabled()
```

NOTE on the adjoint convention: `grammar.Ty('n').l` — check empirically
(`$PY -c "from lambeq.backend.grammar import Ty; print(Ty('n').l.z)"`)
and make `.l` move z by the SAME direction so converters are trivial.
If grammar's `.l` gives z=+1 instead of -1, flip the test's last assert
and the implementation consistently.

- [ ] **Step 2: run, confirm ImportError-style failure.**

`$PY -m pytest tests/backend/test_fast_types.py -q` -> fails (no module).

- [ ] **Step 3: implement.**

`lambeq/backend/fast/validate.py`:

```python
# (Apache header as in other lambeq files)
"""Opt-in validation switch for the fast diagram core."""

from __future__ import annotations

import contextlib
import os

_enabled: bool = os.environ.get('LAMBEQ_FAST_VALIDATE', '') == '1'


def set_validation(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def validation_enabled() -> bool:
    return _enabled


@contextlib.contextmanager
def validation():
    global _enabled
    old, _enabled = _enabled, True
    try:
        yield
    finally:
        _enabled = old


@contextlib.contextmanager
def no_validation():
    global _enabled
    old, _enabled = _enabled, False
    try:
        yield
    finally:
        _enabled = old
```

`lambeq/backend/fast/types.py`:

```python
# (Apache header)
"""Interned atomic types and tuple-based composite types."""

from __future__ import annotations

from collections.abc import Iterator

_IDS: dict[tuple[str, int], int] = {}
_NAMES: list[str] = []
_ZS: list[int] = []
_L: list[int] = []      # atom id -> id of left adjoint (-1 = not built)
_R: list[int] = []


def atom(name: str, z: int = 0) -> int:
    """Intern (name, z) and return its atom id."""
    try:
        return _IDS[name, z]
    except KeyError:
        i = len(_NAMES)
        _IDS[name, z] = i
        _NAMES.append(name)
        _ZS.append(z)
        _L.append(-1)
        _R.append(-1)
        return i


def atom_name(i: int) -> str:
    return _NAMES[i]


def atom_z(i: int) -> int:
    return _ZS[i]


def atom_l(i: int) -> int:
    j = _L[i]
    if j < 0:
        j = atom(_NAMES[i], _ZS[i] - 1)
        _L[i] = j
        _R[j] = i
    return j


def atom_r(i: int) -> int:
    j = _R[i]
    if j < 0:
        j = atom(_NAMES[i], _ZS[i] + 1)
        _R[i] = j
        _L[j] = i
    return j


class FTy:
    """Immutable composite type: a tuple of atom ids with stored hash."""

    __slots__ = ('atoms', '_hash')

    def __init__(self, atoms: tuple[int, ...] = ()) -> None:
        self.atoms = atoms
        self._hash = hash(atoms)

    @classmethod
    def of(cls, *names: str) -> FTy:
        return cls(tuple(atom(name) for name in names))

    def __matmul__(self, other: FTy) -> FTy:
        return FTy(self.atoms + other.atoms)

    @property
    def l(self) -> FTy:
        return FTy(tuple(atom_l(a) for a in reversed(self.atoms)))

    @property
    def r(self) -> FTy:
        return FTy(tuple(atom_r(a) for a in reversed(self.atoms)))

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self) -> Iterator[int]:
        return iter(self.atoms)

    def __getitem__(self, index: int | slice) -> int | FTy:
        if isinstance(index, slice):
            return FTy(self.atoms[index])
        return self.atoms[index]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FTy) and self.atoms == other.atoms

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        if not self.atoms:
            return 'FTy()'
        parts = []
        for a in self.atoms:
            z = _ZS[a]
            suffix = '.l' * -z if z < 0 else '.r' * z
            parts.append(f'{_NAMES[a]}{suffix}')
        return ' @ '.join(parts)
```

(Adjust `atom_l`/`atom_r` z-direction per the Step-1 convention check.)

`lambeq/backend/fast/__init__.py`:

```python
# (Apache header)
"""Fast immutable diagram core. See the 2026-06-12 design spec."""

from lambeq.backend.fast.types import atom, atom_name, atom_z, FTy
from lambeq.backend.fast.validate import (no_validation, set_validation,
                                          validation, validation_enabled)

__all__ = ['atom', 'atom_name', 'atom_z', 'FTy', 'no_validation',
           'set_validation', 'validation', 'validation_enabled']
```

- [ ] **Step 4: run to green** (4 passed), **Step 5: commit**

```bash
git add lambeq/backend/fast tests/backend
git commit -m "Add fast diagram core: interned types and validation switch

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Boxes and diagrams (`diagram.py`)

**Files:** Create `lambeq/backend/fast/diagram.py`; modify
`lambeq/backend/fast/__init__.py` (re-export `FBox`, `FDiagram`, kind
constants); Test `tests/backend/test_fast_diagram.py`.

- [ ] **Step 1: failing tests:**

```python
import pytest

from lambeq.backend import fast
from lambeq.backend.fast import FBox, FDiagram, FTy
from lambeq.backend.fast.diagram import (CAP, CUP, PLAIN, SPIDER, SWAP,
                                         WORD, cap, caps, cup, cups,
                                         spider, swap, word)


def _n():
    return FTy.of('n')


def test_box_identity_and_hash():
    f = FBox('f', _n(), _n() @ _n())
    g = FBox('f', _n(), _n() @ _n())
    assert f == g and hash(f) == hash(g)
    assert f != FBox('g', _n(), _n() @ _n())
    assert f.kind == PLAIN


def test_compose_and_tensor():
    n = _n()
    f = FBox('f', n, n).to_diagram()
    g = FBox('g', n, n).to_diagram()
    d = f >> g
    assert d.dom == n and d.cod == n
    assert [b.name for b, _ in d.terms] == ['f', 'g']
    t = f @ g
    assert t.dom == n @ n and t.cod == n @ n
    assert [o for _, o in t.terms] == [0, 1]


def test_then_validation():
    n, s = FTy.of('n'), FTy.of('s')
    f = FBox('f', n, n).to_diagram()
    h = FBox('h', s, s).to_diagram()
    with pytest.raises(ValueError):
        f >> h
    with fast.no_validation():
        d = f >> h          # trusted mode: no check, garbage in garbage out
        assert d.cod == s


def test_identity_unit_laws():
    n = _n()
    f = FBox('f', n, n).to_diagram()
    assert (FDiagram.id(n) >> f).terms == f.terms
    assert (f @ FDiagram.id(FTy())).terms == f.terms


def test_cups_shape():
    n2 = _n() @ _n()
    d = cups(n2, n2.r)
    assert d.dom == n2 @ n2.r and d.cod == FTy()
    assert all(b.kind == CUP for b, _ in d.terms)
    assert len(d.terms) == 2


def test_caps_shape():
    n2 = _n() @ _n()
    d = caps(n2, n2.r)
    assert d.dom == FTy() and d.cod == n2 @ n2.r
    assert all(b.kind == CAP for b, _ in d.terms)
    assert len(d.terms) == 2


def test_dagger_round_trip():
    n = _n()
    f = FBox('f', n, n @ n).to_diagram()
    dd = f.dagger().dagger()
    assert dd.dom == f.dom and dd.cod == f.cod
    assert dd == f


def test_diagram_hash_eq_stored():
    n = _n()
    a = FBox('f', n, n).to_diagram() >> FBox('g', n, n).to_diagram()
    b = FBox('f', n, n).to_diagram() >> FBox('g', n, n).to_diagram()
    assert a == b and hash(a) == hash(b)
```

- [ ] **Step 2: run (ImportError), Step 3: implement.**

`lambeq/backend/fast/diagram.py`:

```python
# (Apache header)
"""Immutable boxes and term-presentation diagrams."""

from __future__ import annotations

from typing import Any

from lambeq.backend.fast.validate import validation_enabled
from lambeq.backend.fast.types import FTy

PLAIN, WORD, CUP, CAP, SWAP, SPIDER = range(6)
_KIND_NAMES = ('PLAIN', 'WORD', 'CUP', 'CAP', 'SWAP', 'SPIDER')


class FBox:
    """Immutable box. `payload` carries data/symbols by reference and
    participates in equality by identity (or both-None) but not hash."""

    __slots__ = ('name', 'dom', 'cod', 'kind', 'z', 'is_dagger',
                 'payload', '_hash')

    def __init__(self, name: str, dom: FTy, cod: FTy, kind: int = PLAIN,
                 z: int = 0, is_dagger: bool = False,
                 payload: Any = None) -> None:
        self.name = name
        self.dom = dom
        self.cod = cod
        self.kind = kind
        self.z = z
        self.is_dagger = is_dagger
        self.payload = payload
        self._hash = hash((name, dom.atoms, cod.atoms, kind, z, is_dagger))

    def to_diagram(self) -> FDiagram:
        return FDiagram(self.dom, ((self, 0),), self.cod)

    def dagger(self) -> FBox:
        return FBox(self.name, self.cod, self.dom, self.kind, self.z,
                    not self.is_dagger, self.payload)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, FBox)
                and self._hash == other._hash
                and self.name == other.name
                and self.dom == other.dom
                and self.cod == other.cod
                and self.kind == other.kind
                and self.z == other.z
                and self.is_dagger == other.is_dagger
                and (self.payload is other.payload
                     or (self.payload is None and other.payload is None)))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return (f'FBox({self.name!r}, {self.dom!r}, {self.cod!r}, '
                f'kind={_KIND_NAMES[self.kind]})')


class FDiagram:
    """Immutable diagram: dom + ((box, offset), ...) + stored cod/hash."""

    __slots__ = ('dom', 'terms', 'cod', '_hash')

    def __init__(self, dom: FTy, terms: tuple[tuple[FBox, int], ...],
                 cod: FTy) -> None:
        self.dom = dom
        self.terms = terms
        self.cod = cod
        self._hash = hash((dom.atoms, terms, cod.atoms))
        if validation_enabled():
            self._check()

    def _check(self) -> None:
        frontier = list(self.dom.atoms)
        for t, (box, off) in enumerate(self.terms):
            if off < 0 or off + len(box.dom) > len(frontier):
                raise ValueError(
                    f'term {t} ({box.name!r}): offset {off} with dom '
                    f'width {len(box.dom)} does not fit frontier of '
                    f'width {len(frontier)}')
            if tuple(frontier[off:off + len(box.dom)]) != box.dom.atoms:
                raise ValueError(
                    f'term {t} ({box.name!r}) at offset {off}: expected '
                    f'dom {box.dom!r}, frontier has '
                    f'{FTy(tuple(frontier[off:off + len(box.dom)]))!r}')
            frontier[off:off + len(box.dom)] = box.cod.atoms
        if tuple(frontier) != self.cod.atoms:
            raise ValueError(
                f'stored cod {self.cod!r} does not match computed '
                f'{FTy(tuple(frontier))!r}')

    @classmethod
    def id(cls, dom: FTy = FTy()) -> FDiagram:
        return cls(dom, (), dom)

    def then(self, other: FDiagram) -> FDiagram:
        if validation_enabled() and self.cod != other.dom:
            raise ValueError(f'cannot compose: cod {self.cod!r} != '
                             f'dom {other.dom!r}')
        return FDiagram(self.dom, self.terms + other.terms, other.cod)

    __rshift__ = then

    def tensor(self, other: FDiagram) -> FDiagram:
        # While self's terms run, other's wires (width len(other.dom))
        # sit untouched to the RIGHT of self's frontier, so self's
        # offsets are unchanged. After self finishes, its cod is fixed,
        # so other's offsets shift by len(self.cod).
        shift = len(self.cod)
        terms = self.terms + tuple((b, o + shift) for b, o in other.terms)
        return FDiagram(self.dom @ other.dom, terms, self.cod @ other.cod)

    def __matmul__(self, other: FDiagram) -> FDiagram:
        return self.tensor(other)

    def dagger(self) -> FDiagram:
        # Reversing term order and daggering each box keeps every
        # offset valid: a box's offset is the same number of wires from
        # the left whether read top-down or bottom-up.
        new_terms = tuple((box.dagger(), off)
                          for box, off in reversed(self.terms))
        return FDiagram(self.cod, new_terms, self.dom)

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(o for _, o in self.terms)

    @property
    def boxes(self) -> tuple[FBox, ...]:
        return tuple(b for b, _ in self.terms)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, FDiagram)
                and self._hash == other._hash
                and self.dom == other.dom
                and self.terms == other.terms
                and self.cod == other.cod)

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return (f'FDiagram({self.dom!r}, {len(self.terms)} terms, '
                f'{self.cod!r})')


def word(name: str, cod: FTy, payload: Any = None) -> FBox:
    return FBox(name, FTy(), cod, WORD, payload=payload)


def cup(left: FTy, right: FTy) -> FBox:
    if validation_enabled() and left.r != right:
        raise ValueError(f'cup: {right!r} is not the right adjoint '
                         f'of {left!r}')
    return FBox('CUP', left @ right, FTy(), CUP)


def cap(left: FTy, right: FTy) -> FBox:
    if validation_enabled() and left.r != right:
        raise ValueError(f'cap: {right!r} is not the right adjoint '
                         f'of {left!r}')
    return FBox('CAP', FTy(), left @ right, CAP)


def swap(left: FTy, right: FTy) -> FBox:
    return FBox('SWAP', left @ right, right @ left, SWAP)


def spider(ty: FTy, n_legs_in: int, n_legs_out: int) -> FBox:
    dom = FTy(ty.atoms * n_legs_in)
    cod = FTy(ty.atoms * n_legs_out)
    return FBox('SPIDER', dom, cod, SPIDER)


def cups(left: FTy, right: FTy) -> FDiagram:
    """Nested cups contracting left @ right to the empty type."""
    if validation_enabled() and left.r != right:
        raise ValueError(f'cups: {right!r} is not the right adjoint '
                         f'of {left!r}')
    n = len(left)
    terms = tuple(
        (cup(FTy((left.atoms[i],)), FTy((right.atoms[n - 1 - i],))),
         i)
        for i in range(n - 1, -1, -1))
    return FDiagram(left @ right, terms, FTy())


def caps(left: FTy, right: FTy) -> FDiagram:
    """Nested caps producing left @ right from the empty type."""
    if validation_enabled() and left.r != right:
        raise ValueError(f'caps: {right!r} is not the right adjoint '
                         f'of {left!r}')
    n = len(left)
    terms = tuple(
        (cap(FTy((left.atoms[i],)), FTy((right.atoms[n - 1 - i],))),
         i)
        for i in range(n))   # outermost first: grows outward from empty
    return FDiagram(FTy(), terms, left @ right)
```

IMPORTANT implementation notes:
- The `cups` nesting order (innermost first, offsets n-1, n-2, ... 0)
  and the `caps` order (outermost first, offsets 0, 1, ... n-1) must
  match `grammar.Diagram.cups`/`.caps` — Task 3's round-trip test is
  the gate; if it fails on ordering, match the oracle
  (`grammar.py:801`), do not weaken the test.
- No `rotate` in v1: no fast-core consumer (convert/functor/normal/
  contraction) calls it; the spec's "mirroring the grammar API surface
  used by consumers" qualifier makes omission correct. Add it only if
  a later task actually needs it.
- Re-export `FBox`, `FDiagram`, the kind constants and the constructor
  helpers (`word`, `cup`, `cap`, `swap`, `spider`, `cups`, `caps`)
  from `lambeq/backend/fast/__init__.py`.

- [ ] **Step 4: run to green (7 passed), Step 5: commit**

```bash
git add lambeq/backend/fast tests/backend
git commit -m "Add fast diagram core boxes and term-presentation diagrams

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Converters + round-trip gate + `to_fast_diagram`

**Files:** Create `lambeq/backend/fast/convert.py`; modify
`lambeq/backend/fast/__init__.py`, `lambeq/text2diagram/ccg_tree.py`;
Test `tests/backend/test_fast_convert.py`.

**Contract for `convert.py`:**

```python
def ty_to_fast(ty: grammar.Ty) -> FTy            # atoms via atom(name, t.z)
def ty_to_grammar(fty: FTy) -> grammar.Ty        # grammar.Ty(name, z=z), tensored
def box_to_fast(box: grammar.Box) -> FBox
def box_to_grammar(fbox: FBox) -> grammar.Box   # cup/cap types come from fbox.dom/cod themselves
def to_fast(d: grammar.Diagram) -> FDiagram
def to_grammar(d: FDiagram) -> grammar.Diagram
```

- `to_fast`: one pass over `d.layers`; offset = `len(layer.left)`;
  kind by isinstance against `grammar.Cup/Cap/Swap/Spider/Word/
  Daggered` (Daggered -> underlying box converted with
  `is_dagger=True`); `grammar.Frame` raises
  `NotImplementedError('frames are not supported by the fast core')`.
  `box.z` carries over. Payload: `getattr(box, 'data', None)` plus, for
  tensor boxes, keep the box's symbol-bearing data object by reference.
- `to_grammar`: walk terms with an atom-id frontier; for each term
  build the grammar box (dispatch on kind: `grammar.Cup(l, r)`,
  `grammar.Cap(l, r)`, `grammar.Swap(l, r)`, `grammar.Spider` with leg
  counts, `grammar.Word(name, cod)`, plain `grammar.Box(name, dom,
  cod)`; apply `.dagger()` if `is_dagger`) and a
  `grammar.Layer(left_ty, box, right_ty)` from the frontier slices;
  assemble `grammar.Diagram(dom, cod, layers)`.
- `tensor.Dim` handling: Dim atoms intern as `atom(f'#{int}', 0)` with
  the integer stored in a side table `dim_of(atom_id) -> int`
  (`register_dim(n) -> atom_id` helper in `types.py` or `convert.py`);
  `ty_to_fast` on a `tensor.Dim` uses it, `ty_to_grammar` rebuilds
  `Dim(ints...)`. Needed by Tasks 6-7.
- `CCGTree.to_fast_diagram()`: mirror the recursion of
  `to_diagram`/`_to_diagram` (`ccg_tree.py:431+`) but building
  FDiagram/FBox directly. SIMPLEST correct v1: implement as
  `convert.to_fast(self.to_diagram())` first, get the gate green, then
  optimize to the direct recursion ONLY if Task 7's benchmark misses
  the 0.3ms target — record which variant shipped.

- [ ] **Step 1: failing tests** (`tests/backend/test_fast_convert.py`):

```python
import pytest

from lambeq.backend import grammar
from lambeq.backend.fast import FTy, convert


def test_ty_round_trip():
    n, s = grammar.Ty('n'), grammar.Ty('s')
    t = n @ s.l @ n.r
    assert convert.ty_to_grammar(convert.ty_to_fast(t)) == t


def test_simple_diagram_round_trip():
    n, s = grammar.Ty('n'), grammar.Ty('s')
    alice = grammar.Word('Alice', n)
    likes = grammar.Word('likes', n.r @ s @ n.l)
    bob = grammar.Word('Bob', n)
    d = grammar.Diagram.create_pregroup_diagram(
        words=[alice, likes, bob],
        morphisms=[(grammar.Cup, 3, 4), (grammar.Cup, 0, 1)])
    rt = convert.to_grammar(convert.to_fast(d))
    assert rt == d


def test_cups_match_grammar():
    from lambeq.backend.fast.diagram import cups
    n = grammar.Ty('n') @ grammar.Ty('s')
    g = grammar.Diagram.cups(n, n.r)
    f = cups(convert.ty_to_fast(n), convert.ty_to_fast(n.r))
    assert convert.to_grammar(f) == g


def test_frame_raises():
    pytest.importorskip('lambeq')
    n = grammar.Ty('n')
    frame = grammar.Frame('fr', n, n,
                          components=[grammar.Word('w', n)])
    with pytest.raises(NotImplementedError):
        convert.to_fast(frame.to_diagram())


def test_corpus_round_trip(bobcat_diagrams):
    for d in bobcat_diagrams:
        assert convert.to_grammar(convert.to_fast(d)) == d


def test_to_fast_diagram_equivalent(bobcat_trees):
    for t in bobcat_trees:
        assert (convert.to_grammar(t.to_fast_diagram())
                == t.to_diagram())
```

Add to `tests/backend/conftest.py` (module-scoped, 40 captions —
enough signal, bounded runtime; the FULL corpus gate runs in Task 7's
benchmark script):

```python
@pytest.fixture(scope='module')
def bobcat_trees():
    from lambeq import BobcatParser, VerbosityLevel
    parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                          parser_backend='rust')
    with open('/tmp/coco_bench.txt') as f:
        sents = [line.split() for line in f if line.strip()][:40]
    trees = parser.sentences2trees(sents, tokenised=True,
                                   suppress_exceptions=True)
    return [t for t in trees if t is not None]


@pytest.fixture(scope='module')
def bobcat_diagrams(bobcat_trees):
    return [t.to_diagram() for t in bobcat_trees]
```

(Regenerate `/tmp/coco_bench.txt` if missing:
`$PY -c "import json,random,re; anns=json.load(open('/home/kinianlo/projects/discoviz/data/coco/annotations/captions_val2017.json'))['annotations']; TOK=re.compile(r\"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[.,!?;:]\"); caps=[]; seen=set()\nfor a in anns:\n    t=TOK.findall(a['caption'])\n    if 4<=len(t)<=60:\n        l=' '.join(t)\n        if l.lower() not in seen: seen.add(l.lower()); caps.append(l)\nrandom.Random(0).shuffle(caps); open('/tmp/coco_bench.txt','w').write('\\n'.join(caps[:2000])+'\\n')"` —
or copy the equivalent snippet from the Tier-3 history.)

- [ ] **Step 2: fail, Step 3: implement per the contract, Step 4: green**

`$PY -m pytest tests/backend/ -q` (600000ms; the corpus fixture loads
the Bobcat model). Expected: all pass. The corpus round-trip is THE
gate of this task — debug converter mismatches by diffing layer reprs
of the first failing diagram.

- [ ] **Step 5: commit**

```bash
git add lambeq/backend/fast tests/backend lambeq/text2diagram/ccg_tree.py
git commit -m "Add fast/grammar converters with corpus round-trip gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Functor (`functor.py`)

**Files:** Create `lambeq/backend/fast/functor.py`; modify
`lambeq/backend/fast/__init__.py`; Test
`tests/backend/test_fast_functor.py`.

**Contract** (oracle: `grammar.Functor`, `grammar.py:1976-2105`, minus
the copying and double-validation):

```python
class FFunctor:
    def __init__(self, ob, ar):
        # ob: dict[int (atom id), FTy] OR callable (FFunctor, int) -> FTy
        # ar: callable (FFunctor, FBox) -> FBox | FDiagram
    def __call__(self, x: FTy | FBox | FDiagram) -> ...
```

- `ob_atom(a)` caches per atom id (plain dict, returns shared FTy).
- `map_box(box)` caches per FBox (stored-int hash); the cached value is
  returned AS IS — immutability makes sharing safe. Under validation
  (only), check the image's dom/cod equal the mapped dom/cod once.
- Structural boxes are mapped automatically (the oracle's behaviour):
  CUP/CAP -> `cups`/`caps`-style FDiagram over the mapped wire type
  (one cup per mapped atom, same nesting as `cups()`), SWAP -> swaps of
  the mapped types (a swap box over the mapped FTys), SPIDER -> spider
  with the mapped type. Only PLAIN/WORD boxes go through `ar`.
- `map_diagram(d)`: single pass; keep `widths: list[int]` = mapped
  width per frontier wire; for each term: `shift = sum(widths[:off])`
  (running prefix; fine in v1), splice the image's terms shifted by
  `shift`, then update `frontier`/`widths` with the box's cod atoms.

- [ ] **Step 1: failing tests:**

```python
from lambeq.backend import grammar
from lambeq.backend.fast import FFunctor, FTy, convert
from lambeq.backend.fast.types import atom


def test_identity_functor_is_noop(bobcat_diagrams):
    ident = FFunctor(ob=lambda f, a: FTy((a,)),
                     ar=lambda f, b: b)
    for d in (convert.to_fast(g) for g in bobcat_diagrams[:10]):
        assert ident(d) == d


def test_functor_matches_grammar_oracle(bobcat_diagrams):
    """Wire-doubling functor: n -> n@n, s -> s, words double cod."""
    n = grammar.Ty('n')

    def g_ob(_, ty):
        return ty @ ty if ty == n else ty

    def g_ar(functor, box):
        return grammar.Box(box.name, functor(box.dom), functor(box.cod))

    g_functor = grammar.Functor(grammar.grammar, ob=g_ob, ar=g_ar)

    n_id = atom('n')

    def f_ob(_, a):
        return FTy((a, a)) if a == n_id else FTy((a,))

    def f_ar(functor, box):
        from lambeq.backend.fast import FBox
        return FBox(box.name, functor(box.dom), functor(box.cod),
                    box.kind, box.z, box.is_dagger, box.payload)

    f_functor = FFunctor(ob=f_ob, ar=f_ar)
    for g in bobcat_diagrams[:10]:
        expected = g_functor(g)
        got = convert.to_grammar(f_functor(convert.to_fast(g)))
        assert got == expected, g


def test_functor_cache_returns_shared_object():
    from lambeq.backend.fast import FBox
    calls = []

    def ar(functor, box):
        calls.append(box.name)
        return FBox(box.name.upper(), box.dom, box.cod)

    f = FFunctor(ob=lambda _, a: FTy((a,)), ar=ar)
    b = FBox('f', FTy.of('n'), FTy.of('n'))
    d = b.to_diagram() >> b.to_diagram()
    out = f(d)
    assert calls == ['f']                       # second occurrence cached
    assert out.terms[0][0] is out.terms[1][0]   # SHARED object, no copy
```

- [ ] **Step 2: fail, Step 3: implement, Step 4: green, Step 5: commit**

```bash
git add lambeq/backend/fast tests/backend
git commit -m "Add copy-free functor to the fast diagram core

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

NOTE: the oracle test exercises cups/caps/swap auto-mapping through
real Bobcat diagrams (they contain cups). If the grammar oracle maps a
structural box differently than assumed, match the oracle —
`grammar.Functor.__call__` on Cup/Cap dispatches are the truth.

---

### Task 5: Cup/snake removal and normal form (`normal.py`)

**Files:** Create `lambeq/backend/fast/normal.py`; modify `__init__.py`;
Test `tests/backend/test_fast_normal.py`.

**Contract** (oracle: `lambeq/backend/snake_removal.py` — read fully;
also `grammar.Diagram.remove_snakes`/`normal_form` callers in
`grammar.py`):

- `remove_snakes(d: FDiagram) -> FDiagram`: port `snake_removal`'s
  find-snake/unsnake loop onto the terms array. `follow_wire`'s
  obstruction bookkeeping translates to offset comparisons;
  `interchange(i, j)` becomes index arithmetic on a `list[(FBox, int)]`
  working copy (mutate the working list, build one FDiagram at the
  end — interior states need no FDiagram objects at all, unlike the
  oracle which allocates a Diagram per interchange).
- `normal_form(d: FDiagram) -> FDiagram`: port the `normalize()`
  bubble pass the same way.
- Semantics gate: `to_grammar(remove_snakes(to_fast(d)))` must equal
  the oracle's output `grammar.Diagram.remove_snakes(d)` (check the
  exact public name on grammar.Diagram — grep; it may be exposed via
  `normal_form` with `snake_removal` internal) for every corpus
  diagram. Determinism holds because the port follows the same
  scan order.

- [ ] **Step 1: failing tests:**

```python
from lambeq.backend.fast import convert
from lambeq.backend.fast.normal import normal_form, remove_snakes


def test_remove_snakes_matches_oracle(bobcat_diagrams):
    for g in bobcat_diagrams:
        oracle = g.remove_snakes()   # ADJUST to the actual public name
        fast = convert.to_grammar(remove_snakes(convert.to_fast(g)))
        assert fast == oracle, g


def test_normal_form_matches_oracle(bobcat_diagrams):
    for g in bobcat_diagrams[:15]:
        oracle = g.normal_form()
        fast = convert.to_grammar(normal_form(convert.to_fast(g)))
        assert fast == oracle, g
```

FIRST ACTION of this task: grep `grammar.py` and `snake_removal.py`
for the actual public entry points (`normal_form`, `remove_snakes`,
`snake_removal(...)`) and fix the test to call the real oracle API.
Then proceed TDD as usual.

- [ ] **Step 2-4: fail / implement / green** (these are the hardest
ports of the plan; debug by shrinking to the smallest failing diagram
and tracing both implementations' interchange sequences).

- [ ] **Step 5: commit**

```bash
git add lambeq/backend/fast tests/backend
git commit -m "Port snake removal and normal form to the fast core

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Contraction extraction (`contraction.py`) + model integration (`model.py`)

**Files:** Create `lambeq/backend/fast/contraction.py`,
`lambeq/backend/fast/model.py`; modify `__init__.py`; Test
`tests/backend/test_fast_contraction.py`.

**Contract:**

```python
@dataclass(frozen=True)
class ContractionSpec:
    factors: tuple[tuple[Any, tuple[int, ...]], ...]
    #        (payload, index-ids per leg)  — payload: Symbol list/array ref
    out_indices: tuple[int, ...]
    sizes: dict[int, int]          # index id -> dimension


def to_contraction(d: FDiagram) -> ContractionSpec
def evaluate(spec: ContractionSpec, weights: dict, backend='torch') -> Tensor
```

- `to_contraction` walks terms with a frontier of index ids
  (union-find for identifications): dom wires get fresh ids; WORD/PLAIN
  boxes append a factor with (dom ids + fresh cod ids); CUP unifies its
  two wire ids (no factor); CAP creates one fresh id used for both cod
  wires; SWAP permutes frontier ids (no factor); SPIDER unifies all its
  leg ids (delta tensor semantics — VERIFY against
  `lambeq/backend/tensor.py` Spider's array semantics before assuming;
  if lambeq's Spider is not a pure delta, emit it as a factor with an
  explicit payload marker instead). Sizes come from Dim atoms
  (`convert.dim_of`).
- `evaluate` resolves payloads through `weights` (Symbol -> tensor) and
  contracts with `torch.einsum` using the sublist (integer-index) API.
  FACT from `lambeq/training/pytorch_model.py:101-104`: model weights
  are FLAT 1-D parameters (`torch.rand(w.size)` where `w.size` is the
  product of the symbol's dimensions) — `evaluate` must reshape each
  resolved weight to the factor's per-leg dims (from `sizes`) before
  the einsum.
- `lambeq/backend/fast/model.py` — the spec's "behind the existing
  model API" requirement (tensor path only; quantum models keep the
  old path):

```python
# (Apache header)
"""PytorchModel drop-in that contracts via cached ContractionSpecs."""

from __future__ import annotations

import torch

from lambeq.backend.fast import contraction, convert
from lambeq.training.pytorch_model import PytorchModel


class FastPytorchModel(PytorchModel):
    """Identical training API; per-step cost is one weight gather +
    einsum per diagram. No deepcopy, no diagram mutation. Specs are
    extracted once per distinct diagram object (keyed by id; a strong
    reference is kept so ids stay valid)."""

    def __init__(self, tn_path_optimizer=None) -> None:
        super().__init__(tn_path_optimizer)
        self._spec_cache: dict[int, tuple[object, object]] = {}

    def _spec_for(self, diagram):
        key = id(diagram)
        hit = self._spec_cache.get(key)
        if hit is None or hit[0] is not diagram:
            spec = contraction.to_contraction(convert.to_fast(diagram))
            self._spec_cache[key] = (diagram, spec)
            return spec
        return hit[1]

    def get_diagram_output(self, diagrams):
        parameters = dict(zip(self.symbols, self.weights))
        return torch.stack([
            contraction.evaluate(self._spec_for(d), parameters)
            for d in diagrams])
```

- [ ] **Step 1: failing test:**

```python
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from lambeq.backend.fast import contraction, convert


def _circuits(diagrams):
    from lambeq import AtomicType, RemoveCupsRewriter, SpiderAnsatz
    from lambeq.backend.tensor import Dim

    ansatz = SpiderAnsatz({t: Dim(4) for t in (
        AtomicType.NOUN, AtomicType.SENTENCE, AtomicType.NOUN_PHRASE,
        AtomicType.PREPOSITIONAL_PHRASE, AtomicType.CONJUNCTION,
        AtomicType.PUNCTUATION)})
    remove_cups = RemoveCupsRewriter()
    return [ansatz(remove_cups(d)) for d in diagrams[:10]]


def test_pipeline_numeric_gate(bobcat_diagrams):
    """Old path (PytorchModel) and fast path produce identical tensors."""
    from lambeq import PytorchModel

    circuits = _circuits(bobcat_diagrams)
    model = PytorchModel.from_diagrams(circuits)
    torch.manual_seed(0)
    model.initialise_weights()
    expected = model.get_diagram_output(circuits)

    weights = dict(zip(model.symbols, model.weights))
    for circ, exp in zip(circuits, expected):
        spec = contraction.to_contraction(convert.to_fast(circ))
        got = contraction.evaluate(spec, weights)
        assert torch.allclose(got, exp, atol=1e-5), circ


def test_fast_model_matches_pytorch_model(bobcat_diagrams):
    """FastPytorchModel is a drop-in: same API, same numbers."""
    from lambeq import PytorchModel
    from lambeq.backend.fast.model import FastPytorchModel

    circuits = _circuits(bobcat_diagrams)
    old = PytorchModel.from_diagrams(circuits)
    torch.manual_seed(0)
    old.initialise_weights()

    new = FastPytorchModel.from_diagrams(circuits)
    new.symbols = old.symbols
    new.weights = old.weights        # share the exact same parameters

    expected = old.get_diagram_output(circuits)
    got = new.get_diagram_output(circuits)
    assert torch.allclose(got, expected, atol=1e-5)
    got2 = new.get_diagram_output(circuits)   # second call: cached specs
    assert torch.allclose(got2, expected, atol=1e-5)
```

NOTE: read `lambeq/training/pytorch_model.py` first — the exact
weight/symbol pairing and output shape conventions must be mirrored
(symbols sorted? weights as ModuleList?). Adjust the test mechanics to
the real API; the ASSERTION (numerically identical tensors) is the
spec gate and must not weaken. If lambeq's spider/cup numeric
conventions surface scaling factors (e.g. Spider normalisation), match
them in `evaluate` — document any discovered convention in the module
docstring.

- [ ] **Step 2-4: fail / implement / green; Step 5: commit**

```bash
git add lambeq/backend/fast tests/backend
git commit -m "Add contraction extraction and FastPytorchModel with numeric gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Fast pipeline benchmark + full-corpus gates + RESULTS

**Files:** Create `benchmarks/fastdiag_bench.py`; modify
`benchmarks/RESULTS.md`.

- [ ] **Step 1: write `benchmarks/fastdiag_bench.py`** — measures, on
the full 2000-caption corpus (parse once with the rust backend, then
time each op over all diagrams, old vs fast):

```
construction:   tree.to_diagram()           vs tree.to_fast_diagram()
remove_cups:    RemoveCupsRewriter()(d)     vs normal.remove_snakes(fd)
ansatz:         SpiderAnsatz functor (old)  vs equivalent FFunctor
copy:           fast_deepcopy(td)           vs fd (shared ref, document)
hash/eq:        hash(td), td == td2         vs hash(fd), fd == fd2
substitution:   deepcopy+mutate per step    vs to_contraction once + evaluate
round-trip gate: to_grammar(to_fast(d)) == d  over ALL corpus diagrams
```

Print a table of ms/diagram old vs fast vs target (targets from the
spec: 0.3 / 0.4 / 0.75 / ~0 / — / one-off+gather). Structure it like
`benchmarks/bobcat_throughput.py` (same conventions, `--num` flag).

- [ ] **Step 2: run** (`$PY benchmarks/fastdiag_bench.py
/tmp/coco_bench.txt`), confirm the full-corpus round-trip gate passes
(printed as `round-trip: N/N identical`), record the table.

- [ ] **Step 3:** append a `## Fast diagram core` section to
`benchmarks/RESULTS.md` with the measured table, gate results, and
honest verdicts vs the spec targets (misses get a one-line profile of
where the time went, per campaign convention).

- [ ] **Step 4: full test sweep + commit**

```bash
$PY -m pytest tests/backend/ -q
$PY -m pytest tests/test_bobcat.py tests/text2diagram/model_based_reader/test_bobcat_parser.py -q   # untouched modules stay green
git add benchmarks/ 
git commit -m "Benchmark the fast diagram core against the five targets

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin fast-diagram-core
```

Report at the end: the five-op table, gate status, which targets were
met/missed, and whether any op still justifies the deferred Rust
kernel.
