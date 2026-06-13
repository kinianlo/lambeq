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
"""PytorchModel drop-in that contracts via cached ContractionSpecs."""

from __future__ import annotations

import torch

from lambeq.backend.fast import contraction, convert
from lambeq.training.pytorch_model import PytorchModel


class FastPytorchModel(PytorchModel):
    """A drop-in :class:`PytorchModel` for the classical pipeline.

    Identical training API; per-step cost is one weight gather + einsum
    per diagram.  No deepcopy, no diagram mutation.  A
    :class:`~lambeq.backend.fast.contraction.ContractionSpec` is
    extracted once per distinct diagram object (keyed by ``id``; a
    strong reference is kept so ids stay valid).  Checkpoint save/load,
    ``initialise_weights`` and ``from_diagrams`` are inherited
    unchanged.

    This targets the real-valued tensor pipeline; quantum models keep
    the original :class:`PytorchModel` contraction path.

    .. note::
        The id-keyed spec cache holds a strong reference to every
        distinct diagram object for the model's lifetime.  This is
        fine for the standard fixed-dataset flow, but a caller that
        regenerates diagram objects each epoch gets no cache hits
        and unbounded cache growth.
    """

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
        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`FastPytorchModel.from_diagrams()`.')
        parameters = dict(zip(self.symbols, self.weights))
        return torch.stack([
            contraction.evaluate(self._spec_for(d), parameters)
            for d in diagrams])
