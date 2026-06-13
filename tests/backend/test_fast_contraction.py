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
