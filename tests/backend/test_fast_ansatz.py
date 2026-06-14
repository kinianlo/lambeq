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
import pytest

from lambeq import AtomicType, SpiderAnsatz
from lambeq.backend import grammar
from lambeq.backend.fast import convert, FSpiderAnsatz
from lambeq.backend.fast.convert import dim_of
from lambeq.backend.fast.diagram import SPIDER
from lambeq.backend.tensor import Dim

N, S = AtomicType.NOUN, AtomicType.SENTENCE
OB = {N: Dim(2), S: Dim(3)}


def _symbols(fd):
    return {(b.payload.name, b.payload.size)
            for b, _ in fd.terms if b.payload is not None}


def _legacy_symbols(diagram):
    return {(s.name, s.size) for s in diagram.free_symbols}


def test_max_order_validation():
    with pytest.raises(ValueError):
        FSpiderAnsatz(OB, max_order=1)


def test_simple_word_symbol_parity():
    g = grammar.Word('w', grammar.Ty('n') @ grammar.Ty('s'))
    legacy = SpiderAnsatz(OB)(g.to_diagram())
    fast = FSpiderAnsatz(OB)(convert.to_fast(g.to_diagram()))
    assert _symbols(fast) == _legacy_symbols(legacy)


def test_wide_cod_splits():
    n, s = grammar.Ty('n'), grammar.Ty('s')
    g = grammar.Word('likes', n @ s @ n)
    legacy = SpiderAnsatz(OB)(g.to_diagram())
    fast = FSpiderAnsatz(OB)(convert.to_fast(g.to_diagram()))
    assert _symbols(fast) == _legacy_symbols(legacy)
    assert any(b.kind == SPIDER for b, _ in fast.terms)


def test_adjoint_atom_same_dim():
    """An ``n.r`` / ``n.l`` wire maps to the same Dim as ``n``, and the
    resulting Dim atoms remain registered (resolvable by ``dim_of``)."""
    n, s = grammar.Ty('n'), grammar.Ty('s')
    # A word whose codomain carries adjoint atoms.
    g = grammar.Word('verb', n.r @ s @ n.l)
    fast = FSpiderAnsatz(OB)(convert.to_fast(g.to_diagram()))

    # Every wire atom in the tensorised diagram must be a registered
    # Dim atom (this fails if adjoint wires became rotated,
    # unregistered Dim atoms).
    dims = []
    for box, _ in fast.terms:
        for a in tuple(box.dom.atoms) + tuple(box.cod.atoms):
            dims.append(dim_of(a))
    # n -> 2, s -> 3: the adjoints n.r / n.l share n's dimension (2).
    assert set(dims) == {2, 3}

    # Symbol parity against the legacy oracle holds for adjoint cods.
    legacy = SpiderAnsatz(OB)(g.to_diagram())
    assert _symbols(fast) == _legacy_symbols(legacy)
