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
"""Snake removal and normal form on the fast term array.

This is a port of :mod:`lambeq.backend.snake_removal` (the oracle) onto
the flat ``terms`` array of :class:`FDiagram`.  The oracle allocates a
new ``Diagram`` for every interchange; here all the bookkeeping happens
on a mutable ``list[(FBox, int)]`` working copy and a *single*
``FDiagram`` is built at the very end.  The scan order is identical to
the oracle, so the output is deterministic and matches the oracle after
``convert.to_grammar``.
"""

from __future__ import annotations

from lambeq.backend.fast.diagram import CAP, CUP, FBox, FDiagram

Term = tuple[FBox, int]


class InterchangerError(Exception):
    """Raised when two connected boxes are asked to commute."""

    def __init__(self, box0: FBox, box1: FBox) -> None:
        super().__init__(f'Boxes {box0} and {box1} do not commute.')


# ---------------------------------------------------------------------------
# Cup/cap detection.
#
# The oracle keys on ``isinstance(box, Cap)`` / ``isinstance(box,
# Cup)``.  Under ``convert`` a grammar ``Cap`` becomes ``kind == CAP``
# and a grammar ``Cup`` becomes ``kind == CUP``; daggering a cup/cap in
# grammar resolves to the *opposite* class, which ``box_to_grammar``
# reproduces from kind + ``is_dagger``.  Matching the round trip
# therefore means keying on the box *shape* among cup/cap kinds: a "cap"
# has an empty dom and a width-2 cod, a "cup" has an empty cod and a
# width-2 dom -- stable under the dagger flag, and excluding look-alike
# words/spiders.
# ---------------------------------------------------------------------------
def _is_cap(box: FBox) -> bool:
    # Width-2 is intentional: atomic cups/caps are always width 2;
    # multi-atom cups don't exist in the fast encoding.
    return (box.kind in (CUP, CAP)
            and len(box.dom) == 0 and len(box.cod) == 2)


def _is_cup(box: FBox) -> bool:
    # Width-2 is intentional: atomic cups/caps are always width 2;
    # multi-atom cups don't exist in the fast encoding.
    return (box.kind in (CUP, CAP)
            and len(box.cod) == 0 and len(box.dom) == 2)


# ---------------------------------------------------------------------------
# Interchange (right moves only, i.e. the oracle's ``left=False`` path).
# ---------------------------------------------------------------------------
def _swap_adjacent(terms: list[Term], p: int) -> None:
    """Swap the adjacent boxes at positions ``p`` and ``p + 1``.

    Mirrors a single adjacent ``interchange`` of the oracle: ``box0``
    (at ``p``) and ``box1`` (at ``p + 1``) trade places.  The offset
    adjustments are the oracle's two commuting cases written directly in
    offset arithmetic (the layer-width algebra collapses to these):

    * box0 right of box1 (``off0 >= off1 + |box1.dom|``): box1 keeps its
      offset; box0 shifts by ``|box1.cod| - |box1.dom|``.
    * box0 left of box1 (``off1 >= off0 + |box0.cod|``): box0 keeps its
      offset; box1 shifts by ``|box0.dom| - |box0.cod|``.
    """
    box0, off0 = terms[p]
    box1, off1 = terms[p + 1]
    if off0 >= off1 + len(box1.dom):            # box0 right of box1
        new0 = (box1, off1)
        new1 = (box0, off0 + len(box1.cod) - len(box1.dom))
    elif off1 >= off0 + len(box0.cod):          # box0 left of box1
        new0 = (box1, off1 + len(box0.dom) - len(box0.cod))
        new1 = (box0, off0)
    else:
        raise InterchangerError(box0, box1)
    terms[p] = new0
    terms[p + 1] = new1


def _interchange(terms: list[Term], i: int, j: int) -> None:
    """Move the box at position ``i`` to position ``j`` in place.

    Decomposes into a run of adjacent swaps, exactly as the oracle's
    recursive ``interchange`` does for ``left=False``.

    Note: trusts callers for index bounds (oracle raises IndexError on
    out-of-range i/j; internal callers track indices identically so
    bounds are always valid when this is reached).
    """
    if i == j:
        return
    if j < i - 1:
        for k in range(i - j):
            _swap_adjacent(terms, i - k - 1)
    elif j > i + 1:
        for k in range(j - i):
            _swap_adjacent(terms, i + k)
    else:                                       # adjacent
        _swap_adjacent(terms, min(i, j))


# ---------------------------------------------------------------------------
# Snake finding and removal.
# ---------------------------------------------------------------------------
def _follow_wire(terms: list[Term],
                 i: int,
                 j: int) -> tuple[int, int, tuple[list[int], list[int]]]:
    """Port of the oracle's ``follow_wire``.

    Given a box index ``i`` and the offset ``j`` of one of its output
    wires, walk down the term array until the wire is consumed.  Returns
    ``(i, j, (left_obstruction, right_obstruction))`` where ``i`` is the
    consuming box (or ``len(terms)`` for the bottom boundary), ``j``
    the wire offset at its bottom end, and the obstruction lists hold
    the box indices passed on the left/right of the followed wire.
    """
    n = len(terms)
    left_obstruction: list[int] = []
    right_obstruction: list[int] = []
    while i < n - 1:
        i += 1
        box, off = terms[i]
        if off <= j < off + len(box.dom):
            return i, j, (left_obstruction, right_obstruction)
        if off <= j:
            j += len(box.cod) - len(box.dom)
            left_obstruction.append(i)
        else:
            right_obstruction.append(i)
    return n, j, (left_obstruction, right_obstruction)


def _find_snake(
    terms: list[Term]
) -> None | tuple[int, int, tuple[list[int], list[int]], bool]:
    """Port of the oracle's ``find_snake``.

    Returns ``(cup, cap, obstructions, left_snake)`` for the first
    yankable cup/cap pair in scan order, or ``None``.
    """
    n = len(terms)
    for cap in range(n):
        cap_box, cap_off = terms[cap]
        if not _is_cap(cap_box):
            continue
        for left_snake, wire in ((True, cap_off), (False, cap_off + 1)):
            cup, wire, obstructions = _follow_wire(terms, cap, wire)
            not_yankable = (cup == n
                            or not _is_cup(terms[cup][0])
                            or (left_snake and terms[cup][1] + 1 != wire)
                            or (not left_snake and terms[cup][1] != wire))
            if not_yankable:
                continue
            return cup, cap, obstructions, left_snake
    return None


def _unsnake(terms: list[Term],
             cup: int,
             cap: int,
             obstructions: tuple[list[int], list[int]],
             left_snake: bool) -> None:
    """Port of the oracle's ``unsnake``, mutating ``terms`` in place.

    Slides the obstruction boxes out from between the cup and cap, then
    deletes the now-adjacent cup/cap pair (indices ``cap`` .. ``cup``).
    """
    left_obstruction, right_obstruction = obstructions
    if left_snake:
        for box in left_obstruction:
            _interchange(terms, box, cap)
            for i, right_box in enumerate(right_obstruction):
                if right_box < box:
                    right_obstruction[i] += 1
            cap += 1
        for box in right_obstruction[::-1]:
            _interchange(terms, box, cup)
            cup -= 1
    else:
        for box in left_obstruction[::-1]:
            _interchange(terms, box, cup)
            for i, right_box in enumerate(right_obstruction):
                if right_box > box:
                    right_obstruction[i] -= 1
            cup -= 1
        for box in right_obstruction:
            _interchange(terms, box, cap)
            cap += 1
    del terms[cap:cup + 1]


def remove_snakes(d: FDiagram) -> FDiagram:
    """Remove every snake from ``d`` (oracle ``Diagram.remove_snakes``).

    A snake is a cup/cap pair forming ``Id @ Cap >> Cup @ Id`` or
    ``Cap @ Id >> Id @ Cup``; each is straightened to an identity wire.
    """
    terms = list(d.terms)
    while True:
        yankable = _find_snake(terms)
        if yankable is None:
            break
        _unsnake(terms, *yankable)
    return FDiagram(d.dom, tuple(terms), d.cod)


def normal_form(d: FDiagram) -> FDiagram:
    """Connected normal form of ``d`` (oracle: ``Diagram.normal_form``).

    Removes snakes, then bubbles boxes with the oracle's ``normalize``
    pass (``left=False``).  Raises :class:`NotImplementedError` if the
    rewrite revisits a state (the oracle's "not connected" case).
    """
    fd = remove_snakes(d)
    terms = list(fd.terms)
    cache: set[tuple[Term, ...]] = set()
    no_more_moves = False
    while not no_more_moves:
        no_more_moves = True
        for i in range(len(terms) - 1):
            box0, off0 = terms[i]
            box1, off1 = terms[i + 1]
            if off0 >= off1 + len(box1.dom):
                _swap_adjacent(terms, i)
                snapshot = tuple(terms)
                if snapshot in cache:
                    raise NotImplementedError('diagram is not connected.')
                cache.add(snapshot)
                no_more_moves = False
    return FDiagram(fd.dom, tuple(terms), fd.cod)
