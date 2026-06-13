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
"""Extract a flat einsum contraction from a fast tensor diagram.

The fast core represents an ansatz circuit (after ``SpiderAnsatz`` +
``RemoveCupsRewriter``) as an :class:`FDiagram` whose boxes carry
``tensor.Symbol`` payloads and whose wires are interned ``Dim`` atoms.
:func:`to_contraction` walks the terms with a union-find frontier of
*index ids* and produces a :class:`ContractionSpec`: a list of factors
(one per tensor box) plus the contraction indices.  :func:`evaluate`
resolves the symbolic factors against a weight table and contracts them.

Numeric conventions discovered from :mod:`lambeq.backend.tensor`
(``Diagram.to_tn``) and matched here:

* **Spider** -> ``tn.CopyNode`` (a generalised delta).  All of a
  spider's legs carry the *same* index, so we union-find them together;
  the shared index becomes an einsum hyperedge (it may appear on three
  or more factors).  No factor is emitted.
* **Cup** -> ``tn.connect`` (an identity contraction).  We union-find
  its two wire ids; no factor is emitted.
* **Swap** -> permutes the two scanned wires; no factor, no new id.
* **Cap** -> ``tn.Node`` with a *non-delta* array (``arr[0] = 1`` and
  ``arr[-1] = 1``; see ``tensor.Cap.__init__``).  A cap is therefore
  emitted as an explicit constant factor, NOT unified.
* **Word / plain Box** -> a factor whose array is ``box.data`` reshaped
  to ``dom.dim + cod.dim`` (see ``tensor.Box.array``).  Model weights
  are flat 1-D parameters, so the reshape recovers the per-leg axes.
* A box's winding ``z`` (conjugation) is ignored: for the real-valued
  classical pipeline the only conjugated boxes ``RemoveCupsRewriter``
  emits are single-leg states, for which ``tensor.Box._conjugate_array``
  reduces to the identity (and multi-leg conjugation cannot be
  represented on flat data by the oracle either).

The diagram's input (``dom``) wires correspond to ``to_tn``'s leading
``CopyNode(2, dim)`` identity nodes, so each initial frontier id is an
output index; :attr:`ContractionSpec.out_indices` is
``dom ids + final frontier ids`` (matching ``inputs + scan``).
"""

from __future__ import annotations

from dataclasses import dataclass
import string
from typing import Any

import numpy as np

from lambeq.backend.fast.convert import dim_of
from lambeq.backend.fast.diagram import CAP, CUP, FDiagram, SPIDER, SWAP
from lambeq.backend.symbol import Symbol

_LETTERS = string.ascii_letters   # 52 single-character einsum subscripts


@dataclass(frozen=True)
class ContractionSpec:
    """A flattened einsum contraction extracted from an FDiagram.

    Attributes
    ----------
    factors : tuple of (payload, leg ids)
        One entry per tensor box.  ``payload`` is the box's
        :class:`Symbol` (resolved through the weight table at evaluation
        time) or a constant array/scalar.  The leg ids are canonical,
        densely numbered index ids, ordered ``dom legs + cod legs``.
    out_indices : tuple of int
        The dangling index ids, ordered ``dom wires + cod wires``.
    sizes : dict
        Index id -> dimension.

    """

    factors: tuple[tuple[Any, tuple[int, ...]], ...]
    out_indices: tuple[int, ...]
    sizes: dict[int, int]


class _UnionFind:
    """Minimal union-find over integer ids with path compression."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def add(self, x: int) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def to_contraction(d: FDiagram) -> ContractionSpec:
    """Extract a :class:`ContractionSpec` from a fast tensor diagram."""
    uf = _UnionFind()
    counter = 0
    raw_sizes: dict[int, int] = {}

    def fresh(dim: int) -> int:
        nonlocal counter
        i = counter
        counter += 1
        uf.add(i)
        raw_sizes[i] = dim
        return i

    input_ids = [fresh(dim_of(a)) for a in d.dom.atoms]
    frontier = list(input_ids)
    factors: list[tuple[Any, list[int]]] = []

    for box, off in d.terms:
        n_dom = len(box.dom)
        dom_ids = frontier[off:off + n_dom]
        kind = box.kind

        if kind == CUP:
            uf.union(dom_ids[0], dom_ids[1])
            frontier[off:off + n_dom] = []
        elif kind == SWAP:
            frontier[off], frontier[off + 1] = (frontier[off + 1],
                                                frontier[off])
        elif kind == SPIDER:
            # CopyNode delta: every leg shares one index.
            rep = dom_ids[0] if dom_ids else fresh(dim_of(box.cod[0]))
            for x in dom_ids[1:]:
                uf.union(rep, x)
            frontier[off:off + n_dom] = [rep] * len(box.cod)
        elif kind == CAP:
            # Non-delta constant array; emit as an explicit factor.
            d0, d1 = dim_of(box.cod[0]), dim_of(box.cod[1])
            ids = [fresh(d0), fresh(d1)]
            arr = np.zeros(d0 * d1, dtype=np.float32)
            arr[0] = 1.0
            arr[-1] = 1.0
            factors.append((arr, ids))
            frontier[off:off + n_dom] = ids
        else:   # WORD / PLAIN: a tensor box with a payload
            cod_ids = [fresh(dim_of(a)) for a in box.cod]
            if box.z % 2:
                # Conjugated box (``tensor.Box`` with odd winding): its
                # array axes map to the wires in reversed order within
                # the dom block and within the cod block.  This mirrors
                # ``tensor.Box._conjugate_array`` (a no-op on values for
                # real weights, a permutation on legs).
                leg_ids = (dom_ids[::-1] + cod_ids[::-1])
            else:
                leg_ids = dom_ids + cod_ids
            factors.append((box.payload, leg_ids))
            frontier[off:off + n_dom] = cod_ids

    out_raw = input_ids + frontier

    # Remap every id through find() and densely renumber.
    canon: dict[int, int] = {}

    def dense(x: int) -> int:
        r = uf.find(x)
        if r not in canon:
            canon[r] = len(canon)
        return canon[r]

    spec_factors = tuple((payload, tuple(dense(i) for i in ids))
                         for payload, ids in factors)
    out_indices = tuple(dense(i) for i in out_raw)
    sizes = {dense(raw): dim for raw, dim in raw_sizes.items()}
    return ContractionSpec(spec_factors, out_indices, sizes)


def _einsum_pair(t1, ids1: list[int], t2, ids2: list[int],
                 keep: set[int]):
    """Contract two operands, keeping only ids in ``keep``.

    Each step relabels its (few) indices to local a-zA-Z subscripts, so
    the global 52-subscript limit of ``torch.einsum`` never applies and
    hyperedges (indices shared by 3+ factors) are handled by retaining
    them in the output.
    """
    import torch

    local: dict[int, str] = {}

    def sub(i: int) -> str:
        if i not in local:
            local[i] = _LETTERS[len(local)]
        return local[i]

    s1 = ''.join(sub(i) for i in ids1)
    s2 = ''.join(sub(i) for i in ids2)
    out_ids = [i for i in dict.fromkeys(ids1 + ids2) if i in keep]
    s_out = ''.join(local[i] for i in out_ids)
    result = torch.einsum(f'{s1},{s2}->{s_out}', t1, t2)
    return result, out_ids


def _contract(factors: list[tuple[Any, list[int]]],
              out_indices: tuple[int, ...]):
    """Greedily contract factors pairwise into the output tensor."""
    import torch

    if not factors:
        return torch.ones((), dtype=torch.float32)

    work = [(t, list(ids)) for t, ids in factors]
    out_set = set(out_indices)

    while len(work) > 1:
        # Pick the pair sharing the most indices (avoids needless
        # outer products); falls back to the first pair otherwise.
        best = (-1, 0, 1)
        for a in range(len(work)):
            sa = set(work[a][1])
            for b in range(a + 1, len(work)):
                shared = len(sa & set(work[b][1]))
                if shared > best[0]:
                    best = (shared, a, b)
        _, a, b = best
        t2, ids2 = work.pop(b)
        t1, ids1 = work.pop(a)
        keep = set(out_set)
        for _, ids in work:
            keep.update(ids)
        work.append(_einsum_pair(t1, ids1, t2, ids2, keep))

    tensor, ids = work[0]
    if list(ids) == list(out_indices):
        return tensor

    if len(set(out_indices)) != len(out_indices):
        raise NotImplementedError(
            'repeated output indices (e.g. a multi-output spider feeding '
            'directly to the diagram codomain) are not supported')

    # Final relabel/sum/transpose into the requested output order.
    local: dict[int, str] = {}

    def sub(i: int) -> str:
        if i not in local:
            local[i] = _LETTERS[len(local)]
        return local[i]

    s_in = ''.join(sub(i) for i in ids)
    s_out = ''.join(sub(i) for i in out_indices)
    return torch.einsum(f'{s_in}->{s_out}', tensor)


def evaluate(spec: ContractionSpec, weights: dict, backend: str = 'torch'):
    """Evaluate a :class:`ContractionSpec` against a weight table.

    Parameters
    ----------
    spec : ContractionSpec
        The contraction to evaluate.
    weights : dict
        Maps each :class:`Symbol` to its (flat) parameter tensor.
    backend : str, default 'torch'
        Only ``'torch'`` is supported.

    """
    if backend != 'torch':
        raise NotImplementedError(f'unsupported backend: {backend!r}')

    import torch

    factors: list[tuple[Any, list[int]]] = []
    for payload, ids in spec.factors:
        dims = tuple(spec.sizes[i] for i in ids)
        if isinstance(payload, Symbol):
            try:
                raw = weights[payload]
            except KeyError:
                raw = payload.scale * weights[payload.unscaled]
        else:
            raw = torch.as_tensor(payload, dtype=torch.float32)
        factors.append((raw.reshape(dims), list(ids)))

    return _contract(factors, spec.out_indices)
