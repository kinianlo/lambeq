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
"""Copy-free box-level rewriter over the fast diagram core.

This mirrors :class:`lambeq.rewrite.Rewriter` (the oracle), which is a
``grammar.Functor(ob=identity, ar=_ar)`` whose ``_ar`` returns the first
matching rule's rewrite or the box unchanged.  We reproduce that exact
behaviour with :class:`~lambeq.backend.fast.functor.FFunctor` (the
copy-free whole-diagram traversal), while REUSING the legacy
:class:`RewriteRule` logic per box: each candidate ``FBox`` is
reconstructed into a faithful ``grammar.Box`` (via
:func:`~lambeq.backend.fast.convert.box_to_grammar`) and handed to the
same ``rule(gbox)`` callables, so matched fragments are exactly the
legacy rewrites.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Union

from lambeq.backend import grammar
from lambeq.backend.fast import convert
from lambeq.backend.fast.diagram import FBox, FDiagram
from lambeq.backend.fast.functor import FFunctor
from lambeq.backend.fast.types import FTy
from lambeq.rewrite import RewriteRule, Rewriter

Image = Union[FBox, FDiagram]


class FRewriter:
    """Box-level rewriter on the fast diagram core.

    A drop-in mirror of :class:`lambeq.rewrite.Rewriter` that operates
    on :class:`FDiagram` values.  It accepts the same rule specification
    (presets by name or :class:`RewriteRule` instances) and applies the
    legacy rules functorially via :class:`FFunctor`.
    """

    def __init__(self,
                 rules: Iterable[RewriteRule | str] | None = None) -> None:
        """Initialise an FRewriter.

        Parameters
        ----------
        rules : iterable of str or RewriteRule, optional
            A list of rewrite rules to use, identical to the argument of
            :class:`lambeq.rewrite.Rewriter`.  ``RewriteRule`` instances
            are used directly, ``str`` objects name default rules.  If
            omitted, all the default rules are used.

        """
        # Reuse the legacy rule resolution (presets, validation, etc.).
        self.rules: list[RewriteRule] = Rewriter(rules).rules
        self.functor = FFunctor(ob=self._ob, ar=self._ar)

    def _ob(self, functor: FFunctor, atom_id: int) -> FTy:
        # Identity on objects, mirroring Rewriter._ob.
        return FTy((atom_id,))

    def _ar(self, functor: FFunctor, box: FBox) -> Image:
        # Reconstruct a faithful grammar box (word/dagger/z preserved)
        # and delegate to the legacy rules, exactly as Rewriter._ar.
        gbox = convert.box_to_grammar(box)
        for rule in self.rules:
            out = rule(gbox)
            if out is not None:
                diagram = (out if isinstance(out, grammar.Diagram)
                           else out.to_diagram())
                return convert.to_fast(diagram)
        return box

    def __call__(self, diagram: FDiagram) -> FDiagram:
        """Apply the rewrite rules to the given fast diagram."""
        return self.functor(diagram)
