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
"""Copy-free functor over the fast diagram core.

This mirrors :class:`lambeq.backend.grammar.Functor` (the oracle) but
drops its two cost centres: the per-call ``fast_deepcopy`` of cached
results and the double type-validation.  Immutability of :class:`FBox`
and :class:`FDiagram` makes cache sharing safe, so cached images are
returned *as is*.

Structural-box behaviour reproduces the oracle's ``apply_functor``
dispatch (grammar.py): rotated atoms/boxes are unwound, mapped, then
rotated back; ``Cup``/``Cap``/``Swap``/``Spider`` are auto-mapped over
the mapped wire type; daggered boxes map the undaggered box and dagger
the image.  Only ``PLAIN``/``WORD`` boxes are handed to the user ``ar``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

from lambeq.backend.fast.diagram import (CAP, CUP, FBox, FDiagram,
                                         SPIDER, SWAP)
from lambeq.backend.fast.types import atom, atom_name, atom_z, FTy
from lambeq.backend.fast.validate import validation_enabled

Image = Union[FBox, FDiagram]


def _rotate_ty(ty: FTy, z: int) -> FTy:
    """Rotate a type by ``z`` adjoints (``.r`` if positive, ``.l`` if
    negative).  Mirrors :meth:`grammar.Ty.rotate`."""
    if z == 0 or not ty.atoms:
        return ty
    out = ty
    if z > 0:
        for _ in range(z):
            out = out.r
    else:
        for _ in range(-z):
            out = out.l
    return out


class FFunctor:
    """A functor on the fast diagram core.

    Parameters
    ----------
    ob : callable or dict
        Either:

        * A callable ``(FFunctor, int) -> FTy`` mapping a *non-rotated*
          atom id (an atom with ``z == 0``) to a type.  Rotated atoms
          are handled automatically by unwinding and rotating
          the result.
        * A ``dict[int, FTy]`` whose keys are ``z == 0`` atom ids (as
          returned by :func:`~lambeq.backend.fast.types.atom`).  The
          dict entries are copied into the internal atom cache at
          construction time.  Atoms not present in the dict map to
          themselves (identity: ``FTy((a,))``).

    ar : callable
        ``(FFunctor, FBox) -> FBox | FDiagram`` mapping a ``PLAIN`` or
        ``WORD`` box (with ``z == 0`` and ``is_dagger == False``) to its
        image.  Structural and rotated/daggered boxes
        never reach ``ar``.
    """

    def __init__(self,
                 ob: Union[Callable[['FFunctor', int], FTy], dict],
                 ar: Callable[['FFunctor', FBox], Image]) -> None:
        self._atom_cache: dict[int, FTy] = {}
        self._box_cache: dict[FBox, Image] = {}
        if isinstance(ob, dict):
            self._atom_cache.update(ob)   # copy; don't alias caller's dict
            self.custom_ob = None
        else:
            self.custom_ob = ob
        self.custom_ar = ar

    # -- dispatch -----------------------------------------------------
    def __call__(self, x: FTy | FBox | FDiagram) -> FTy | Image:
        if isinstance(x, FTy):
            return self.ob(x)
        if isinstance(x, FDiagram):
            return self.map_diagram(x)
        if isinstance(x, FBox):
            return self.map_box(x)
        raise TypeError(f'FFunctor cannot be applied to {type(x)!r}')

    # -- objects ------------------------------------------------------
    def ob_atom(self, a: int) -> FTy:
        """Map a ``z == 0`` atom id, caching the shared result."""
        try:
            return self._atom_cache[a]
        except KeyError:
            ty = (FTy((a,)) if self.custom_ob is None
                  else self.custom_ob(self, a))
            self._atom_cache[a] = ty
            return ty

    def _map_atom(self, a: int) -> FTy:
        z = atom_z(a)
        if z == 0:
            return self.ob_atom(a)
        # Mirror Ty.apply_functor: map the unwound atom, rotate the
        # image by the winding number.
        base = self.ob_atom(atom(atom_name(a), 0))
        return _rotate_ty(base, z)

    def ob(self, ty: FTy) -> FTy:
        out = FTy()
        for a in ty.atoms:
            out = out @ self._map_atom(a)
        return out

    # -- boxes --------------------------------------------------------
    def map_box(self, box: FBox) -> Image:
        """Map a box, caching the image per FBox value.  The cached
        image is shared (returned as is) - safe because images are
        immutable."""
        try:
            return self._box_cache[box]
        except KeyError:
            pass

        image = self._compute_image(box)

        if validation_enabled():
            dom = self.ob(box.dom)
            cod = self.ob(box.cod)
            if image.dom != dom or image.cod != cod:
                raise TypeError(
                    f'FFunctor: image of {box.name!r} is ill-defined: got '
                    f'dom={image.dom!r}, cod={image.cod!r}; expected '
                    f'dom={dom!r}, cod={cod!r}')

        self._box_cache[box] = image
        return image

    def _compute_image(self, box: FBox) -> Image:
        # Daggered box: map the undaggered box, dagger the image.
        if box.is_dagger:
            return self.map_box(box.dagger()).dagger()

        kind = box.kind
        if kind == CUP:
            return self._map_cup(box)
        if kind == CAP:
            return self._map_cap(box)
        if kind == SWAP:
            return self._map_swap(box)
        if kind == SPIDER:
            return self._map_spider(box)

        # PLAIN / WORD.  Rotated boxes are unwound, mapped,
        # rotated back.
        if box.z != 0:
            unwound = FBox(box.name,
                           _rotate_ty(box.dom, -box.z),
                           _rotate_ty(box.cod, -box.z),
                           kind, 0, box.is_dagger, box.payload)
            return _rotate_image(self.map_box(unwound), box.z)
        return self.custom_ar(self, box)

    # -- structural boxes ---------------------------------------------
    def _map_cup(self, box: FBox) -> FDiagram:
        # A grammar cup is atomic: one left wire, one right wire.
        # Note: composite reversed cups (z odd, multi-atom mapped type)
        # are unsupported -- grammar itself raises ValueError for them.
        left = self.ob(box.dom[:1])
        right = self.ob(box.dom[1:])
        z = box.z
        n = len(left)
        # grammar Cup.__new__ order: zip(reversed(left), right),
        # the i-th pair sits at offset n - 1 - i.
        terms = tuple(
            (FBox('CUP', FTy((left[n - 1 - i],)) @ FTy((right[i],)),
                  FTy(), CUP, z), n - 1 - i)
            for i in range(n))
        return FDiagram(left @ right, terms, FTy())

    def _map_cap(self, box: FBox) -> FDiagram:
        left = self.ob(box.cod[:1])
        right = self.ob(box.cod[1:])
        z = box.z
        n = len(left)
        # grammar Cap.__new__ order: zip(left, reversed(right)),
        # the i-th pair sits at offset i.
        terms = tuple(
            (FBox('CAP', FTy(),
                  FTy((left[i],)) @ FTy((right[n - 1 - i],)), CAP, z), i)
            for i in range(n))
        return FDiagram(FTy(), terms, left @ right)

    def _map_swap(self, box: FBox) -> FDiagram:
        left = self.ob(box.dom[:1])
        right = self.ob(box.dom[1:])
        # Mirror grammar Swap.__new__: nested elementary swaps.
        terms = []
        for start in range(len(right)):
            ob = right[start]
            for i in reversed(range(len(left))):
                la = left[i]
                terms.append(
                    (FBox('SWAP', FTy((la,)) @ FTy((ob,)),
                          FTy((ob,)) @ FTy((la,)), SWAP), start + i))
        return FDiagram(left @ right, tuple(terms), right @ left)

    def _map_spider(self, box: FBox) -> FBox:
        src = box.dom.atoms or box.cod.atoms
        if not src:
            raise ValueError('FFunctor: cannot map a spider with no legs')
        n_in, n_out = len(box.dom), len(box.cod)
        typ = self.ob(FTy((src[0],)))
        if len(typ) != 1:
            # grammar builds a composite spider via
            # permutations; this is out of scope for v1
            # (no spiders in the corpus).  Raising is
            # an honest mirror of "unsupported", not a
            # silent wrong answer.
            raise NotImplementedError(
                'FFunctor: spider whose wire type maps to a multi-atom '
                'type is not supported')
        return FBox('SPIDER', FTy(typ.atoms * n_in),
                    FTy(typ.atoms * n_out), SPIDER)

    # -- diagrams -----------------------------------------------------
    def map_diagram(self, d: FDiagram) -> FDiagram:
        widths = [len(self._map_atom(a)) for a in d.dom.atoms]
        new_terms: list[tuple[FBox, int]] = []
        for box, off in d.terms:
            image = self.map_box(box)
            shift = sum(widths[:off])
            if isinstance(image, FDiagram):
                new_terms.extend((b, o + shift) for b, o in image.terms)
            else:
                new_terms.append((image, shift))
            w = len(box.dom)
            widths[off:off + w] = [len(self._map_atom(a))
                                   for a in box.cod.atoms]
        return FDiagram(self.ob(d.dom), tuple(new_terms), self.ob(d.cod))


# ---------------------------------------------------------------------------
# Rotation of an *image* (target-category value), used by
# the z != 0 path.
# Mirrors grammar's Box.rotate / Diagram.rotate / Layer.rotate.
# ---------------------------------------------------------------------------
def _rotate_image(image: Image, z: int) -> Image:
    if isinstance(image, FDiagram):
        return _rotate_fdiagram(image, z)
    return _rotate_fbox(image, z)


def _rotate_fbox(b: FBox, z: int) -> FBox:
    kind = b.kind
    if kind == CUP:
        left, right = b.dom[:1], b.dom[1:]
        if z % 2 == 1:
            left, right = right, left
        left, right = _rotate_ty(left, z), _rotate_ty(right, z)
        return FBox('CUP', left @ right, FTy(), CUP, (b.z + z) % 2,
                    b.is_dagger, b.payload)
    if kind == CAP:
        left, right = b.cod[:1], b.cod[1:]
        if z % 2 == 1:
            left, right = right, left
        left, right = _rotate_ty(left, z), _rotate_ty(right, z)
        return FBox('CAP', FTy(), left @ right, CAP, (b.z + z) % 2,
                    b.is_dagger, b.payload)
    if kind == SWAP:
        left, right = b.dom[:1], b.dom[1:]
        if z % 2 == 1:
            left, right = right, left
        left, right = _rotate_ty(left, z), _rotate_ty(right, z)
        return FBox('SWAP', left @ right, right @ left, SWAP, 0,
                    b.is_dagger, b.payload)
    if kind == SPIDER:
        src = b.dom.atoms or b.cod.atoms
        n_in, n_out = len(b.dom), len(b.cod)
        rot = _rotate_ty(FTy((src[0],)), z)
        return FBox('SPIDER', FTy(rot.atoms * n_in),
                    FTy(rot.atoms * n_out), SPIDER, 0, b.is_dagger, b.payload)
    # PLAIN / WORD: rotate dom/cod, accumulate winding number.
    return FBox(b.name, _rotate_ty(b.dom, z), _rotate_ty(b.cod, z),
                kind, b.z + z, b.is_dagger, b.payload)


def _rotate_fdiagram(d: FDiagram, z: int) -> FDiagram:
    odd = z % 2 == 1
    # Frontier width before each term (in the un-rotated
    # diagram); used to reflect offsets for odd rotations
    # (Layer.rotate swaps left/right).
    widths_before: list[int] = []
    running = len(d.dom)
    for box, _ in d.terms:
        widths_before.append(running)
        running += len(box.cod) - len(box.dom)

    new_terms = []
    for (box, off), width in zip(d.terms, widths_before):
        rb = _rotate_fbox(box, z)
        new_off = width - off - len(box.dom) if odd else off
        new_terms.append((rb, new_off))
    return FDiagram(_rotate_ty(d.dom, z), tuple(new_terms),
                    _rotate_ty(d.cod, z))
