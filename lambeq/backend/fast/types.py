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
    """Immutable composite type: a tuple of atom ids with
    stored hash."""

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
    def l(self) -> FTy:  # noqa: E741, E743
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
