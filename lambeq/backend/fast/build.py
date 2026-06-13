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
"""Fast-core equivalents of the grammar CCG combinators.

Each function takes ``FTy`` arguments and returns an ``FDiagram``,
mirroring ``grammar.Diagram.{fa,ba,fc,bc,fx,bx}`` (grammar.py:769-792)
and the generalised/type-raising fragments in ``CCGRule.apply``
(ccg_rule.py:204-314). Pure fast-core: no text2diagram dependency.
"""

from __future__ import annotations

from lambeq.backend.fast.diagram import CAP, FBox, FDiagram, cups, swap
from lambeq.backend.fast.types import FTy


def _id(ty: FTy) -> FDiagram:
    return FDiagram.id(ty)


def _caps_any(left: FTy, right: FTy) -> FDiagram:
    """Multi-atom caps, bypassing fast-core right-adjoint validation.

    The fast ``caps`` helper requires ``left.r == right``; grammar-style
    type-raising caps need ``left == right.r`` instead (e.g.
    ``caps(T, T.l)``).  Creating FBoxes directly skips that check while
    still passing ``FDiagram._check`` (offsets only, no adjoint check).
    CAP FBoxes convert correctly via ``convert.box_to_grammar``.
    """
    n = len(left)
    terms = tuple(
        (FBox('CAP', FTy(),
              FTy((left.atoms[i], right.atoms[n - 1 - i])), CAP), i)
        for i in range(n)
    )
    return FDiagram(FTy(), terms, left @ right)


def swaps(left: FTy, right: FTy) -> FDiagram:
    """Decomposed complex swap, mirroring grammar.Swap.__new__
    (grammar.py:1853): bubble each right atom left through the left
    block. Matches the oracle's expanded elementary-swap form."""
    nl = len(left)
    frontier = list(left.atoms + right.atoms)
    terms = []
    for start in range(len(right)):
        for i in range(nl - 1, -1, -1):
            off = start + i
            a, b = frontier[off], frontier[off + 1]
            terms.append((swap(FTy((a,)), FTy((b,))), off))
            frontier[off], frontier[off + 1] = b, a
    return FDiagram(left @ right, tuple(terms), right @ left)


def fa(left: FTy, right: FTy) -> FDiagram:
    return _id(left) @ cups(right.l, right)


def ba(left: FTy, right: FTy) -> FDiagram:
    return cups(left, left.r) @ _id(right)


def fc(left: FTy, middle: FTy, right: FTy) -> FDiagram:
    return _id(left) @ cups(middle.l, middle) @ _id(right.l)


def bc(left: FTy, middle: FTy, right: FTy) -> FDiagram:
    return _id(left.r) @ cups(middle, middle.r) @ _id(right)


def fx(left: FTy, middle: FTy, right: FTy) -> FDiagram:
    return (_id(left) @ swaps(middle.l, right.r) @ _id(middle)
            >> swaps(left, right.r) @ cups(middle.l, middle))


def bx(left: FTy, middle: FTy, right: FTy) -> FDiagram:
    return (_id(middle) @ swaps(left.l, middle.r) @ _id(right)
            >> cups(middle, middle.r) @ swaps(left.l, right))


def gfc(left: FTy, middle: FTy, tail: FTy) -> FDiagram:
    return _id(left) @ cups(middle.l, middle) @ _id(tail)


def gbc(prefix: FTy, middle: FTy, right: FTy) -> FDiagram:
    return _id(prefix) @ cups(middle, middle.r) @ _id(right)


def gfx(mid: FTy, left: FTy, join: FTy, right: FTy) -> FDiagram:
    inner = (swaps(mid @ join.l, left) @ _id(join)
             >> _id(left @ mid) @ cups(join.l, join))
    return inner @ _id(right)


def gbx(mid: FTy, left: FTy, join: FTy, right: FTy) -> FDiagram:
    inner = (_id(join) @ swaps(right, join.r @ mid)
             >> cups(join, join.r) @ _id(mid @ right))
    return _id(left) @ inner


def ftr(result: FTy, dom0: FTy) -> FDiagram:
    return _caps_any(result, result.l) @ _id(dom0)


def btr(result: FTy, dom0: FTy) -> FDiagram:
    return _id(dom0) @ _caps_any(result.r, result)
