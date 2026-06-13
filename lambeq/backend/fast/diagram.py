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
"""Immutable boxes and term-presentation diagrams."""

from __future__ import annotations

from typing import Any

from lambeq.backend.fast.validate import validation_enabled
from lambeq.backend.fast.types import FTy

PLAIN, WORD, CUP, CAP, SWAP, SPIDER = range(6)
_KIND_NAMES = ('PLAIN', 'WORD', 'CUP', 'CAP', 'SWAP', 'SPIDER')


class FBox:
    """Immutable box. `payload` carries data/symbols by reference and
    participates in equality by identity (or both-None) but not hash.

    Slots are never reassigned after __init__; _hash is precomputed.
    """

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
    """Immutable diagram: dom + ((box, offset), ...) + stored cod/hash.

    Slots are never reassigned after __init__; _hash is precomputed.
    """

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
            frontier[off:off + len(box.dom)] = list(box.cod.atoms)
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
