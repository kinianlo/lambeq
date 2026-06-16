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
"""One-call fast classical pipeline: trees -> tensorised FDiagrams."""

from __future__ import annotations

from collections.abc import Mapping

from lambeq.backend import grammar
from lambeq.backend.fast.ansatz import FSpiderAnsatz
from lambeq.backend.fast.normal import remove_cups
from lambeq.backend.tensor import Dim


def compile_fast_circuits(trees, ob_map: Mapping[grammar.Ty, Dim],
                          max_order: int = 2):
    """Build tensorised fast FDiagram circuits from CCG trees.

    Equivalent to the legacy ``to_diagram -> RemoveCupsRewriter ->
    SpiderAnsatz`` pipeline, on the fast core.
    """
    ansatz = FSpiderAnsatz(ob_map, max_order)
    return [ansatz(remove_cups(t.to_fast_diagram())) for t in trees]


def compile_quantum_input(trees):
    """Cup-reduced grammar.Diagrams for any CircuitAnsatz, built via the
    fast front-end (fast construction + fast remove_cups).

    Equivalent to ``[RemoveCupsRewriter()(t.to_diagram()) for t in
    trees]`` but built on the fast core, so quantum preprocessing gets
    the construction + cup-removal speedups. The quantum ansatz and
    backends consume the result unchanged.
    """
    # deferred: avoids a circular import via fast/__init__.py
    from lambeq.backend.fast import convert
    return [convert.to_grammar(remove_cups(t.to_fast_diagram()))
            for t in trees]
