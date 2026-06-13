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
