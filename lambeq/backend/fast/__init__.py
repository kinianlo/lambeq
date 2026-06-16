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
"""Fast immutable diagram core. See the 2026-06-12 design spec."""

from lambeq.backend.fast.types import atom, atom_name, atom_z, FTy
from lambeq.backend.fast.validate import (no_validation, set_validation,
                                          validation, validation_enabled)
from lambeq.backend.fast.diagram import (
    CAP, CUP, PLAIN, SPIDER, SWAP, WORD,
    FBox, FDiagram,
    cap, caps, cup, cups, spider, swap, word,
)
from lambeq.backend.fast.functor import FFunctor
from lambeq.backend.fast.normal import (normal_form, remove_cups,
                                        remove_snakes)
from lambeq.backend.fast import build, contraction, convert
from lambeq.backend.fast.ansatz import FSpiderAnsatz
from lambeq.backend.fast.pipeline import (compile_fast_circuits,
                                          compile_quantum_input)

__all__ = ['atom', 'atom_name', 'atom_z', 'FTy', 'no_validation',
           'set_validation', 'validation', 'validation_enabled',
           'CAP', 'CUP', 'PLAIN', 'SPIDER', 'SWAP', 'WORD',
           'FBox', 'FDiagram', 'FFunctor',
           'cap', 'caps', 'cup', 'cups', 'spider', 'swap', 'word',
           'normal_form', 'remove_cups', 'remove_snakes',
           'build', 'contraction', 'convert',
           'FSpiderAnsatz', 'compile_fast_circuits', 'compile_quantum_input']
