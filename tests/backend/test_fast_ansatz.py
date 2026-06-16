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


def test_uncurry_hybrid_path():
    """A wide box with a NON-empty domain (dom+cod > max_order) routes
    through ``_uncurry_hybrid``; its symbols must match the legacy
    oracle.
    """
    n, s = grammar.Ty('n'), grammar.Ty('s')
    # dom=n (len 1), cod=n@s (len 2): dom non-empty and 1+2=3 > max
    # order (2), so this exercises the uncurry fallback, not the
    # simple path.
    box = grammar.Box('verb', n, n @ s)
    assert len(box.dom) and len(box.dom) + len(box.cod) > 2

    diag = box.to_diagram()
    legacy = SpiderAnsatz(OB)(diag)
    fast = FSpiderAnsatz(OB)(convert.to_fast(diag))

    # Uncurrying splits the box into several boxes/spiders: the simple
    # path would emit a single term, so >1 term confirms the branch ran.
    assert len(fast.terms) > 1
    assert _symbols(fast) == _legacy_symbols(legacy)


def test_numeric_pipeline_matches_pytorch_model(bobcat_diagrams):
    assert bobcat_diagrams
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


def test_multi_factor_dim_wire_order():
    """Regression for the multi-factor Dim wire-order bug.

    With ``OB2`` mapping ``n`` to a multi-factor ``Dim(2, 5)``, an
    adjoint wire ``n.r`` must map to ``Dim(5, 2)`` (reversed factor
    order) -- ``Dim(2, 5).rotate(1) == Dim(5, 2)``.  The pre-fix
    ``_map_ty`` dropped the winding and produced ``#2 @ #5``, a silently
    transposed contraction.  Checked both via FFunctor validation (the
    image must be well-defined) and numerically against the legacy
    ``SpiderAnsatz`` through ``PytorchModel``.
    """
    torch = pytest.importorskip('torch')
    from lambeq import PytorchModel
    from lambeq.backend.fast import contraction
    from lambeq.backend.fast.validate import validation

    OB2 = {N: Dim(2, 5), S: Dim(3)}
    n, s = grammar.Ty('n'), grammar.Ty('s')
    g = grammar.Word('w', n.r @ s)
    diag = g.to_diagram()

    legacy = SpiderAnsatz(OB2)(diag)
    # Validation ON: the pre-fix _map_ty makes the box image ill-defined
    # (cod=#2@#5@#3 vs expected #5@#2@#3) and this raises.
    with validation():
        fast = FSpiderAnsatz(OB2)(convert.to_fast(diag))

    # The mapped wire order must match the legacy reversed-factor order.
    fast_box = next(b for b, _ in fast.terms if b.payload is not None)
    assert fast_box.cod == convert.ty_to_fast(legacy.boxes[0].cod)

    assert _symbols(fast) == _legacy_symbols(legacy)

    model = PytorchModel.from_diagrams([legacy])
    torch.manual_seed(0)
    model.initialise_weights()
    expected = model.get_diagram_output([legacy])[0]
    weights = dict(zip(model.symbols, model.weights))

    got = contraction.evaluate(contraction.to_contraction(fast), weights)
    assert torch.allclose(got, expected, atol=1e-5)
