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
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from lambeq.backend import grammar
from lambeq.backend.fast import contraction, convert


def _ansatz_circuits(diagrams):
    from lambeq import AtomicType, RemoveCupsRewriter, SpiderAnsatz
    from lambeq.backend.tensor import Dim

    ansatz = SpiderAnsatz({t: Dim(4) for t in (
        AtomicType.NOUN, AtomicType.SENTENCE, AtomicType.NOUN_PHRASE,
        AtomicType.PREPOSITIONAL_PHRASE, AtomicType.CONJUNCTION,
        AtomicType.PUNCTUATION)})
    remove_cups = RemoveCupsRewriter()
    out = []
    for d in diagrams:
        try:
            out.append(ansatz(remove_cups(d)))
        except Exception:
            pass
    return out


def _circuits(diagrams):
    return _ansatz_circuits(diagrams[:10])


def _tn_oracle(diagram):
    """Contract a concrete (non-symbolic) tensor diagram with the stock
    ``tensornetwork`` path -- the canonical lambeq semantics."""
    import tensornetwork as tn
    from lambeq.backend.numerical_backend import backend

    with backend('pytorch'), tn.DefaultBackend('pytorch'):
        return tn.contractors.auto(*diagram.to_tn()).tensor


@pytest.fixture(scope='module')
def cap_gate_circuits():
    """SpiderAnsatz + RemoveCups circuits over enough corpus sentences to
    include a representative number of cap-containing diagrams.

    The first ~300 COCO sentences yield ~44 cap circuits; the stock
    ``diagrams[:10]`` slice the old gate used contained zero caps.
    """
    from lambeq import BobcatParser, VerbosityLevel

    parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                          parser_backend='rust')
    with open('/tmp/coco_bench.txt') as f:
        sents = [line.split() for line in f if line.strip()][:300]
    trees = parser.sentences2trees(sents, tokenised=True,
                                   suppress_exceptions=True)
    diagrams = [t.to_diagram() for t in trees if t is not None]
    return _ansatz_circuits(diagrams)


def _has_cap(circ):
    return any(isinstance(b, grammar.Cap) for b in circ.boxes)


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


def test_cap_corpus_numeric_gate(cap_gate_circuits):
    """Broadened gate: the fast path matches the stock contraction for
    EVERY corpus circuit, INCLUDING cap-containing ones.

    The stock ``PytorchModel`` can now contract cap circuits thanks to the
    ``tensor.Diagram.to_tn`` cap-dtype fix, so it is a direct oracle here.
    """
    from lambeq import PytorchModel

    circuits = cap_gate_circuits
    n_caps = sum(_has_cap(c) for c in circuits)
    assert n_caps >= 30, f'gate not representative: only {n_caps} cap circuits'

    model = PytorchModel.from_diagrams(circuits)
    torch.manual_seed(0)
    model.initialise_weights()
    weights = dict(zip(model.symbols, model.weights))

    for circ in circuits:
        exp = model.get_diagram_output([circ])[0]
        spec = contraction.to_contraction(convert.to_fast(circ))
        got = contraction.evaluate(spec, weights)
        assert torch.allclose(got, exp, atol=1e-5), circ


@pytest.mark.parametrize('z', [0, 1])
@pytest.mark.parametrize('is_dagger', [False, True])
def test_conjugate_adjoint_cases(z, is_dagger):
    """Every (z-parity, is_dagger) combination of a multi-leg box wired to
    distinct caps must match the canonical semantics exactly.

    The daggered, both-blocks-non-empty cases (the genuine adjoint
    block-swap) are what the old leg-reversal hack transposed.
    """
    from lambeq.backend.tensor import Box, Cap, Dim, Id

    # Uniform leg dims: lambeq's tensor conjugation reverses leg order
    # within a block, so distinct leg dims are ill-formed for odd z even
    # in the stock oracle.  The data is still asymmetric, so a wrong
    # permutation (e.g. a missing adjoint block-swap) is detected.
    rng = np.random.default_rng(0)
    box = Box('B', Dim(2), Dim(2, 2),
              data=rng.standard_normal(8).astype(np.float32), z=z)
    if is_dagger:
        # B.dagger(): dom Dim(2, 2), cod Dim(2) -- both inner blocks
        # non-empty, so the adjoint genuinely block-swaps.
        bridge = box.dagger()
        diagram = (Cap(Dim(2), Dim(2))
                   >> (Id(Dim(2, 2)) @ Cap(Dim(2), Dim(2)))
                   >> (Id(Dim(2)) @ bridge @ Id(Dim(2))))
    else:
        diagram = Cap(Dim(2), Dim(2)) >> (box @ Id(Dim(2)))

    expected = _tn_oracle(diagram)
    spec = contraction.to_contraction(convert.to_fast(diagram))
    got = contraction.evaluate(spec, {})
    assert torch.allclose(got, expected, atol=1e-5), (
        f'z={z} is_dagger={is_dagger} maxdiff='
        f'{(expected - got).abs().max().item()}')


def test_cup_branch():
    """CUP branch of to_contraction: unions two frontier index ids, emits no
    factor, and shrinks the output shape below the uncontracted product.

    The CUP branch is never exercised by the corpus gate (SpiderAnsatz +
    RemoveCupsRewriter eliminates all cups before they reach the fast core).
    Here we build a concrete-array tensor diagram directly:

      W : Dim(1) -> Dim(2, 2, 3)   (3-legged box with float32 data)
      Cup(Dim(2), Dim(2)) @ Id(Dim(3))

    Without cup unification the output would have shape (2, 2, 3); with it
    the two Dim(2) legs are aliased to the SAME index and summed, yielding
    shape (3,).  The numeric value must match the tensornetwork oracle.
    """
    from lambeq.backend.tensor import Box, Cup, Dim, Id

    rng = np.random.default_rng(42)
    box = Box('W', Dim(1), Dim(2, 2, 3),
              data=rng.standard_normal(12).astype(np.float32))
    # Cup contracts the first two Dim(2) cod legs; the Dim(3) wire survives.
    diagram = box >> (Cup(Dim(2), Dim(2)) @ Id(Dim(3)))

    expected = _tn_oracle(diagram)
    spec = contraction.to_contraction(convert.to_fast(diagram))

    # Verify the cup actually unified an index: the factor's first two leg
    # ids must be identical (both mapped to the same canonical index).
    assert len(spec.factors) == 1, 'expected exactly one tensor factor'
    leg_ids = spec.factors[0][1]
    assert leg_ids[0] == leg_ids[1], (
        f'CUP did not unify the two Dim(2) indices: {leg_ids}')

    got = contraction.evaluate(spec, {})
    # Output is (3,), smaller than the uncontracted product (2*2*3 = 12).
    assert got.shape == (3,), f'expected shape (3,), got {got.shape}'
    assert torch.allclose(got, expected, atol=1e-5), (
        f'CUP contraction mismatch: max_diff='
        f'{(got - expected).abs().max().item():.2e}')


def test_daggered_cap_contracts_like_cup():
    """Daggered CAP (kind=CAP, empty cod, width-2 dom) contracts
    like a CUP — union-find the two wires, no cap factor emitted.

    FBox.dagger() keeps ``kind`` and swaps dom/cod, so a CAP FBox
    daggered yields kind=CAP, dom=a@b, cod=() — the mirror of the
    already-handled daggered-CUP case (kind=CUP, dom=(), cod=a@b).
    This path is not reached by the normal tensor backend
    (Cap.dagger() returns a grammar Cup which converts to kind=CUP),
    but must be handled for direct fast-layer use and for symmetry.

    Concretely: a WORD box W: () -> Dim(2)@Dim(2) with data = identity
    matrix, followed by a daggered CAP consuming those two wires, should
    produce the same scalar as W followed by a real CUP (both compute
    sum_i W[i,i] = trace = 2).
    """
    from lambeq.backend.fast.diagram import FBox, FDiagram, CAP, CUP, WORD
    from lambeq.backend.fast.types import FTy
    from lambeq.backend.fast.convert import register_dim

    a_atom = register_dim(2)
    a_ty = FTy((a_atom,))
    ab_ty = a_ty @ a_ty

    # Concrete 2×2 identity matrix as the word payload.
    data = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    word_box = FBox('W_dcap', FTy(), ab_ty, WORD, payload=data)

    # Build a daggered CAP via FBox.dagger() (keeps kind=CAP,
    # swaps dom/cod).
    cap_fbox = FBox('CAP', FTy(), ab_ty, CAP)
    daggered_cap = cap_fbox.dagger()

    # Verify the box is genuinely a daggered-cap shape.
    assert daggered_cap.kind == CAP, (
        f'expected kind=CAP ({CAP}), got {daggered_cap.kind}')
    assert len(daggered_cap.cod) == 0, (
        f'expected empty cod, got len={len(daggered_cap.cod)}')
    assert len(daggered_cap.dom) == 2, (
        f'expected dom width 2, got {len(daggered_cap.dom)}')

    # Diagrams: word → daggered_cap  vs  word → CUP
    dag_cap_diagram = FDiagram(
        FTy(), ((word_box, 0), (daggered_cap, 0)), FTy())
    cup_box = FBox('CUP', ab_ty, FTy(), CUP)
    cup_diagram = FDiagram(
        FTy(), ((word_box, 0), (cup_box, 0)), FTy())

    dag_cap_spec = contraction.to_contraction(dag_cap_diagram)
    cup_spec = contraction.to_contraction(cup_diagram)

    # Daggered CAP must emit NO cap factor (only the word box).
    assert len(dag_cap_spec.factors) == 1, (
        f'daggered-cap should produce exactly 1 factor (the word box), '
        f'got {len(dag_cap_spec.factors)}')

    got = contraction.evaluate(dag_cap_spec, {})
    expected = contraction.evaluate(cup_spec, {})
    assert torch.allclose(got, expected, atol=1e-5), (
        f'daggered-cap mismatch: got {got}, expected {expected}')


def test_nested_cap_daggered_box_repro():
    """Minimal deterministic repro: two nested caps with a daggered
    multi-leg box bridging them.  The old code returned the transpose."""
    from lambeq.backend.tensor import Box, Cap, Dim, Id

    rng = np.random.default_rng(7)
    # Daggered box with both inner blocks non-empty -> genuine block-swap.
    box = Box('W', Dim(2), Dim(2, 2),
              data=rng.standard_normal(8).astype(np.float32))
    diagram = (Cap(Dim(2), Dim(2))
               >> (Id(Dim(2)) @ Cap(Dim(2), Dim(2)) @ Id(Dim(2)))
               >> (box.dagger() @ Id(Dim(2)) @ Id(Dim(2))))

    expected = _tn_oracle(diagram)
    spec = contraction.to_contraction(convert.to_fast(diagram))
    got = contraction.evaluate(spec, {})
    assert torch.allclose(got, expected, atol=1e-5), (
        (expected - got).abs().max().item())
