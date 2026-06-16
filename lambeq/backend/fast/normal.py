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

from lambeq.backend.fast.diagram import (CAP, CUP, cups as _fast_cups,
                                         FBox, FDiagram, PLAIN)
from lambeq.backend.fast.types import FTy

Term = tuple[FBox, int]

# Marker name for a compressed (multi-wire) cup, identical to the legacy
# ``RemoveCupsRewriter``'s ``CUP_TOKEN`` so the round trip reproduces
# the oracle's ``Box(CUP_TOKEN, dom, Ty())`` byte for byte.
CUP_TOKEN = '**CUP**'


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


# ---------------------------------------------------------------------------
# Cup removal: a port of the ``RemoveCupsRewriter`` oracle in
# ``lambeq.rewrite.rewrite_diagram``.
#
# The oracle works on grammar ``Diagram`` fragments; this port keeps the
# exact same control flow but on ``FDiagram`` fragments built with the
# fast ops.  The three rich grammar operations the oracle relies on --
# diagram ``dagger``, single-step rotation (``.r``/``.l``) and
# ``Diagram.cups`` -- are reproduced below so the result round-trips
# byte-identically through ``convert.to_grammar``.
# ---------------------------------------------------------------------------
def _then_at(diag: FDiagram, box: FDiagram, off: int) -> FDiagram:
    """``Diagram.then_at`` for the fast core (no native equivalent)."""
    cod = diag.cod
    left = FDiagram.id(cod[:off])
    right = FDiagram.id(cod[off + len(box.dom):])
    return diag >> (left @ box @ right)


def _rotate_box(box: FBox, z: int) -> FBox:
    """Rotate a single box, mirroring grammar ``Box.rotate``.

    Daggered / cup boxes round-trip correctly: ``box_to_grammar``
    rebuilds them from kind + ``z`` (mod 2 for cups, full ``z`` for
    words), and the generic ``z + z`` update preserves both readings.
    """
    new_dom = box.dom.r if z == 1 else box.dom.l
    new_cod = box.cod.r if z == 1 else box.cod.l
    return FBox(box.name, new_dom, new_cod, box.kind,
                box.z + z, box.is_dagger, box.payload)


def _rotate(d: FDiagram, z: int) -> FDiagram:
    """Single-step diagram rotation (``z == +1``/``-1``).

    Mirrors grammar ``Diagram.rotate``: an odd rotation flips the wire
    order, so a box at ``off`` in a frontier of width ``W`` lands at
    ``W - off - |dom|``; layer order is preserved.
    """
    terms = []
    width = len(d.dom)
    for box, off in d.terms:
        new_off = width - off - len(box.dom) if z % 2 else off
        terms.append((_rotate_box(box, z), new_off))
        width += len(box.cod) - len(box.dom)
    new_dom = d.dom.r if z == 1 else d.dom.l
    new_cod = d.cod.r if z == 1 else d.cod.l
    return FDiagram(new_dom, tuple(terms), new_cod)


def _cups(left: FTy, right: FTy, is_reversed: bool) -> FDiagram:
    """``Diagram.cups(left, right, is_reversed)`` for the fast core.

    Normal (non-reversed) cups -- including the composite cups produced
    by the compression pass -- are exactly ``fast.diagram.cups``.
    Reversed cups only ever arise atomically (compression never builds a
    reversed composite cup), so each atomic pair becomes a reversed cup
    box (``z`` odd), matching grammar's ``Cup(..., is_reversed=True)``.
    """
    if not is_reversed:
        return _fast_cups(left, right)
    if len(left) > 1:
        raise ValueError(
            'Reversed composite cups are not supported; '
            'grammar\'s Diagram.cups raises for this case too.')
    n = len(left)
    terms = tuple(
        (FBox('CUP',
              FTy((left.atoms[i],)) @ FTy((right.atoms[n - 1 - i],)),
              FTy(), CUP, z=1),
         i)
        for i in range(n - 1, -1, -1))
    return FDiagram(left @ right, terms, FTy())


def _compress_cups(d: FDiagram) -> FDiagram:
    """Merge each adjacent nested cup pair into a ``CUP_TOKEN`` box."""
    layers: list[Term] = []
    for box, offset in d.terms:
        nested = (_is_cup(box)
                  and layers
                  and _is_cup(layers[-1][0])
                  and offset == layers[-1][1] - 1)
        if nested:
            prev_box, _ = layers[-1]
            dom = box.dom[:1] @ prev_box.dom @ box.dom[1:]
            layers[-1] = (FBox(CUP_TOKEN, dom, FTy(), PLAIN), offset)
        else:
            layers.append((box, offset))

    diag = FDiagram.id(d.dom)
    for box, offset in layers:
        diag = _then_at(diag, box.to_diagram(), offset)
    return diag


def _remove_cups_pass(d: FDiagram) -> FDiagram:
    """One greedy contraction pass (oracle ``_remove_cups``)."""
    diags: list[FDiagram] = [FDiagram.id(d.dom)]
    for box, offset in d.terms:
        i = 0
        off = offset
        # find the first fragment the offset lands in
        while i < len(diags) and off >= len(diags[i].cod):
            off -= len(diags[i].cod)
            i += 1
        if off == 0 and len(box.dom) == 0:
            diags.insert(i, box.to_diagram())
        else:
            left = diags[i]
            right = FDiagram.id()
            j = 1
            # extend the right fragment until it is wide enough
            while len(left.cod) + len(right.cod) < off + len(box.dom):
                right = right @ diags[i + j]
                j += 1

            cod = left.cod @ right.cod
            wires_l = FDiagram.id(cod[:off])
            wires_r = FDiagram.id(cod[off + len(box.dom):])
            if box.name == CUP_TOKEN or _is_cup(box):
                pg_len = len(box.dom) // 2
                pg_type1 = box.dom[:pg_len]
                pg_type2 = box.dom[pg_len:]
                if len(left.cod) == pg_len and len(left.dom) == 0:
                    if pg_type1.r == pg_type2:
                        new_diag = right >> (_rotate(left.dagger(), 1)
                                             @ wires_r)
                    else:  # illegal (reversed) cup
                        new_diag = right >> (_rotate(left.dagger(), -1)
                                             @ wires_r)
                elif len(right.cod) == pg_len and len(right.dom) == 0:
                    if pg_type1.r == pg_type2:
                        new_diag = left >> (wires_l
                                            @ _rotate(right.dagger(), -1))
                    else:
                        new_diag = left >> (wires_l
                                            @ _rotate(right.dagger(), 1))
                else:
                    nbox = _cups(pg_type1, pg_type2,
                                 pg_type2 != pg_type1.r)
                    new_diag = (left @ right) >> (wires_l @ nbox @ wires_r)
            else:
                new_diag = ((left @ right)
                            >> (wires_l @ box.to_diagram() @ wires_r))
            diags[i:i + j] = [new_diag]

    result = FDiagram.id()
    for dg in diags:
        result = result @ dg
    return result


def remove_cups(d: FDiagram) -> FDiagram:
    """Remove cups from ``d`` (oracle ``RemoveCupsRewriter``).

    Fewer cups means fewer post-selections downstream.  Equivalent to
    ``_remove_cups(_compress_cups(_remove_cups(d)))``.
    """
    return _remove_cups_pass(_compress_cups(_remove_cups_pass(d)))
