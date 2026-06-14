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
"""FFunctor-based SpiderAnsatz over the fast diagram core.

:class:`FSpiderAnsatz` is a pure-Python, copy-free re-implementation of
:class:`lambeq.ansatz.SpiderAnsatz` built on :class:`FFunctor`.  It maps
a pregroup :class:`FDiagram` to a tensorised :class:`FDiagram` whose
wires are interned ``Dim`` atoms and whose boxes carry
:class:`lambeq.backend.symbol.Symbol` payloads, and is **numerically
identical** to the legacy ansatz (gated against ``PytorchModel`` in
Task 2).  The legacy :class:`SpiderAnsatz` is left untouched as the
oracle.

Two semantics-preserving facts make the FFunctor route exact:

* ``FFunctor`` unwinds windings and daggers *before* dispatching to the
  user ``ar`` (mirroring grammar's :meth:`Box.apply_functor` /
  :meth:`Daggered.apply_functor`), so symbol names and directed
  dom/cod -- which the legacy ansatz also computes on the unwound,
  undaggered box -- match exactly.
* The tensor category is *rotation-invariant* (``Dim.rotate`` is the
  identity).  FFunctor's generic rigid rotation would otherwise send an
  adjoint pregroup wire (e.g. ``n.r``) to a *rotated* Dim atom that the
  contraction's ``dim_of`` table does not know about.  We therefore mark
  every registered Dim atom as **self-dual** (its left and right
  adjoints are itself), which turns ``_rotate_ty`` into a no-op on Dim
  atoms and keeps every wire a registered ``z == 0`` Dim atom.
"""

from __future__ import annotations

from collections.abc import Mapping

from lambeq.backend import grammar
from lambeq.backend.fast import convert
from lambeq.backend.fast.convert import register_dim, ty_to_grammar
from lambeq.backend.fast.diagram import (FBox, FDiagram, spider, WORD)
from lambeq.backend.fast.functor import FFunctor
from lambeq.backend.fast.types import atom_name, FTy
from lambeq.backend.symbol import Symbol
from lambeq.backend.tensor import Dim


class FSpiderAnsatz:
    """Split large boxes into spiders, over the fast diagram core.

    Parameters
    ----------
    ob_map : Mapping[grammar.Ty, Dim]
        A mapping from atomic :class:`grammar.Ty` to the ``Dim`` it uses
        in the tensor network.
    max_order : int, default 2
        The maximum order of each tensor, which must be at least 2.

    """

    def __init__(self,
                 ob_map: Mapping[grammar.Ty, Dim],
                 max_order: int = 2) -> None:
        if max_order < 2:
            raise ValueError('`max_order` must be at least 2')

        self.ob_map = ob_map
        self.max_order = max_order
        # Base-name -> total dimension, for directed dom/cod products
        # (adjoints share their base's dim: Dim.rotate is identity).
        self._dim_of_name = {ty.name: int(dim.product)
                             for ty, dim in ob_map.items()}
        self.functor = FFunctor(ob=self._ob, ar=self._ar)

    # -- objects ------------------------------------------------------
    def _ob(self, functor: FFunctor, atom_id: int) -> FTy:
        """Map a ``z == 0`` pregroup atom to its Dim type.

        FFunctor only ever calls this for unwound (``z == 0``) atoms;
        adjoints are unwound to their base, so the base name is the
        correct ``ob_map`` key.  Each Dim atom is marked self-dual so
        that FFunctor's rotation of adjoint wires is a no-op.
        """
        name = atom_name(atom_id)
        dim = self.ob_map[grammar.Ty(name)]
        # register_dim marks every Dim atom self-dual, so FFunctor's
        # rotation of adjoint wires is a no-op per factor.
        return FTy(tuple(register_dim(d) for d in dim.dim))

    def _map_ty(self, fty: FTy) -> FTy:
        """Dim-map a pregroup type via the functor's own object map.

        Delegating to ``self.functor.ob`` honours each atom's winding:
        an adjoint wire (e.g. ``n.r``) maps to the *rotated* image of
        its base, which reverses the factor order of a multi-factor
        ``Dim`` (``Dim(2, 5).rotate(1) == Dim(5, 2)``).  Concatenating
        per-atom ``_ob`` images instead would drop the winding and
        transpose the wires.
        """
        return self.functor.ob(fty)

    # -- arrows -------------------------------------------------------
    def _ar(self, functor: FFunctor, box: FBox) -> FBox | FDiagram:
        """Map a PLAIN/WORD box (already unwound and undaggered)."""
        if len(box.dom) + len(box.cod) <= self.max_order:
            return self._sym_box(f'{box.name}_0', box.dom, box.cod,
                                 box.kind)
        if len(box.dom):
            # Wide box with a non-empty domain: legacy uncurries it.
            # This does not arise for pregroup (state) diagrams; defer
            # to the legacy ansatz for an exact, low-risk split.
            return self._uncurry_hybrid(box)
        # Wide codomain, empty domain: split into a chain of spiders.
        return self._spider_chain(box)

    # -- symbol-carrying leaf box -------------------------------------
    def _sym_box(self, name: str, fdom: FTy, fcod: FTy,
                 kind: int) -> FDiagram:
        sym = self._symbol(name, fdom, fcod)
        return FBox(name, self._map_ty(fdom), self._map_ty(fcod),
                    kind, 0, False, sym).to_diagram()

    def _symbol(self, name: str, fdom: FTy, fcod: FTy) -> Symbol:
        """Reproduce the legacy ansatz's Symbol for a box exactly."""
        from lambeq.ansatz.base import BaseAnsatz

        gdom = ty_to_grammar(fdom)
        gcod = ty_to_grammar(fcod)
        # Call the legacy summariser directly for byte-identical names.
        sym_name = BaseAnsatz._summarise_box(grammar.Box(name, gdom, gcod))
        dd, dc = self._directed_products(gdom, gcod)
        return Symbol(sym_name, directed_dom=dd, directed_cod=dc)

    def _directed_products(self, gdom: grammar.Ty,
                           gcod: grammar.Ty) -> tuple[int, int]:
        """Port of ``TensorAnsatz._generate_directed_dom_cod``
        (tensor.py:68-109) returning the integer dom/cod products.

        Cod atoms with even winding flow to the cod, odd to the dom; dom
        atoms with even winding flow to the dom, odd to the cod.
        Adjoints share their base dimension.
        """
        directed_dom = 1
        directed_cod = 1
        for ty in gcod:
            d = self._dim_of_name[ty.name]
            if ty.z % 2:
                directed_dom *= d
            else:
                directed_cod *= d
        for ty in gdom:
            d = self._dim_of_name[ty.name]
            if ty.z % 2:
                directed_cod *= d
            else:
                directed_dom *= d
        return directed_dom, directed_cod

    # -- wide-codomain spider chain (port of _split_ar, tensor.py) ----
    def _spider_chain(self, box: FBox) -> FDiagram:
        fcod = box.cod
        step = self.max_order - 1

        word_diags: list[FDiagram] = []
        spider_diags = [FDiagram.id(self._map_ty(fcod[:1]))]
        for i, start in enumerate(range(0, len(fcod) - 1, step)):
            cod_slice = fcod[start:start + step + 1]
            word_diags.append(
                self._sym_box(f'{box.name}_{i}', FTy(), cod_slice, WORD))
            spider_diags.append(FDiagram.id(self._map_ty(cod_slice[1:-1])))
            spider_diags.append(
                spider(self._map_ty(cod_slice[-1:]), 2, 1).to_diagram())
        # The last slice's final leg is a codomain output, not a
        # contraction: replace its spider with an identity.
        spider_diags[-1] = FDiagram.id(spider_diags[-1].cod)

        words = FDiagram.id()
        for w in word_diags:
            words = words @ w
        chain = FDiagram.id()
        for s in spider_diags:
            chain = chain @ s
        return words >> chain

    # -- uncurry fallback (rare; defer to the legacy oracle) ----------
    def _uncurry_hybrid(self, box: FBox) -> FDiagram:
        from lambeq.ansatz.tensor import SpiderAnsatz

        legacy = SpiderAnsatz(self.ob_map, self.max_order)
        gbox = grammar.Box(box.name,
                           ty_to_grammar(box.dom),
                           ty_to_grammar(box.cod))
        fragment = legacy._split_ar(None, gbox)
        if isinstance(fragment, grammar.Box):
            fragment = fragment.to_diagram()
        tensorised = legacy.functor(fragment)
        return convert.to_fast(tensorised)

    # -- entry point --------------------------------------------------
    def __call__(self, fdiagram: FDiagram) -> FDiagram:
        return self.functor(fdiagram)
