# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Shims between the fast core and the grammar/tensor oracles."""

from __future__ import annotations

from lambeq.backend import grammar
from lambeq.backend.fast.diagram import (CAP, CUP, FBox, FDiagram, PLAIN,
                                         SPIDER, SWAP, WORD)
from lambeq.backend.fast.types import (atom, atom_name, atom_z, FTy)


# ---------------------------------------------------------------------------
# tensor.Dim side table: Dim atoms intern as ``#<int>`` with the integer
# kept here so the round-trip can rebuild an exact ``Dim``.
# ---------------------------------------------------------------------------
_DIM_OF: dict[int, int] = {}


def register_dim(n: int) -> int:
    """Intern a tensor dimension and return its atom id."""
    a = atom(f'#{n}', 0)
    _DIM_OF[a] = n
    return a


def dim_of(atom_id: int) -> int:
    """Return the integer dimension stored for a Dim atom id."""
    return _DIM_OF[atom_id]


def _is_dim(ty: grammar.Ty) -> bool:
    from lambeq.backend import tensor
    return isinstance(ty, tensor.Dim)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
def ty_to_fast(ty: grammar.Ty) -> FTy:
    """Convert a grammar/tensor type to an :class:`FTy`."""
    if _is_dim(ty):
        return FTy(tuple(register_dim(int(t.name)) for t in ty))
    return FTy(tuple(atom(t.name, t.z) for t in ty))


def ty_to_grammar(fty: FTy) -> grammar.Ty:
    """Convert an :class:`FTy` back to a grammar/tensor type."""
    if len(fty) and fty[0] in _DIM_OF:
        from lambeq.backend import tensor
        return tensor.Dim(*(dim_of(a) for a in fty))
    return grammar.Ty._fromiter(
        grammar.Ty(atom_name(a), z=atom_z(a)) for a in fty)


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------
def box_to_fast(box: grammar.Box) -> FBox:
    """Convert a grammar box to an :class:`FBox`."""
    if isinstance(box, grammar.Frame):
        raise NotImplementedError(
            'frames are not supported by the fast core')
    if isinstance(box, grammar.Daggered):
        # Dagger keeps the kind, swaps dom/cod and flips the flag.
        return box_to_fast(box.box).dagger()

    if isinstance(box, grammar.Cup):
        kind = CUP
    elif isinstance(box, grammar.Cap):
        kind = CAP
    elif isinstance(box, grammar.Swap):
        kind = SWAP
    elif isinstance(box, grammar.Spider):
        kind = SPIDER
    elif isinstance(box, grammar.Word):
        kind = WORD
    else:
        kind = PLAIN

    return FBox(box.name,
                ty_to_fast(box.dom),
                ty_to_fast(box.cod),
                kind,
                box.z,
                False,
                getattr(box, 'data', None))


def box_to_grammar(fbox: FBox) -> grammar.Box:
    """Convert an :class:`FBox` back to a grammar box.

    Dispatch on kind FIRST, build the un-daggered grammar box, then
    apply ``.dagger()`` if the FBox carries ``is_dagger``.
    """
    # Orient to the un-daggered box's dom/cod.
    if fbox.is_dagger:
        f_dom, f_cod = fbox.cod, fbox.dom
    else:
        f_dom, f_cod = fbox.dom, fbox.cod

    kind = fbox.kind
    if kind == CUP:
        box: grammar.Box = grammar.Cup(ty_to_grammar(f_dom[:1]),
                                       ty_to_grammar(f_dom[1:]),
                                       is_reversed=bool(fbox.z % 2))
    elif kind == CAP:
        box = grammar.Cap(ty_to_grammar(f_cod[:1]),
                          ty_to_grammar(f_cod[1:]),
                          is_reversed=bool(fbox.z % 2))
    elif kind == SWAP:
        box = grammar.Swap(ty_to_grammar(f_dom[:1]),
                           ty_to_grammar(f_dom[1:]))
    elif kind == SPIDER:
        atoms = f_dom.atoms or f_cod.atoms
        if not atoms:
            raise ValueError('cannot convert a spider with no legs')
        typ = ty_to_grammar(FTy((atoms[0],)))
        box = grammar.Spider(typ, len(f_dom), len(f_cod))
    elif kind == WORD:
        box = grammar.Word(fbox.name, ty_to_grammar(f_cod), z=fbox.z)
    else:  # PLAIN
        box = grammar.Box(fbox.name,
                          ty_to_grammar(f_dom),
                          ty_to_grammar(f_cod),
                          z=fbox.z)

    if fbox.is_dagger:
        box = box.dagger()
    return box


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------
def to_fast(d: grammar.Diagram) -> FDiagram:
    """Convert a grammar diagram to an :class:`FDiagram` in one pass."""
    terms = tuple((box_to_fast(layer.box), len(layer.left))
                  for layer in d.layers)
    return FDiagram(ty_to_fast(d.dom), terms, ty_to_fast(d.cod))


def rs_to_fast(rsdiagram) -> FDiagram:
    """Materialise an FDiagram from a bobcat_rs RsDiagram export.

    The Rust atom ids are re-interned here via atom(name, z), so the
    result carries Python atom ids."""
    def _fty(atoms):
        return FTy(tuple(atom(name, z) for name, z in atoms))

    dom_atoms, terms, cod_atoms = rsdiagram.export()
    fterms = []
    for name, b_dom, b_cod, kind, z, is_dagger, off in terms:
        box = FBox(name, _fty(b_dom), _fty(b_cod), kind, z, bool(is_dagger))
        fterms.append((box, int(off)))
    return FDiagram(_fty(dom_atoms), tuple(fterms), _fty(cod_atoms))


def to_grammar(d: FDiagram) -> grammar.Diagram:
    """Convert an :class:`FDiagram` back to a grammar diagram.

    Note: PLAIN and WORD boxes are rebuilt as plain ``grammar.Box`` /
    ``grammar.Word`` objects — the original ``.data`` / payload is NOT
    restored.  The round trip is therefore lossless for grammar-category
    diagrams only; tensor pipelines should read :class:`FBox` payloads
    directly rather than converting back to grammar.
    """
    frontier = list(d.dom.atoms)
    layers = []
    for fbox, off in d.terms:
        box = box_to_grammar(fbox)
        width = len(fbox.dom)
        left = ty_to_grammar(FTy(tuple(frontier[:off])))
        right = ty_to_grammar(FTy(tuple(frontier[off + width:])))
        layers.append(grammar.Layer(left, box, right))
        frontier[off:off + width] = list(fbox.cod.atoms)
    return grammar.Diagram(dom=ty_to_grammar(d.dom),
                           cod=ty_to_grammar(d.cod),
                           layers=layers)
