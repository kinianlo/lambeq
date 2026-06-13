#!/usr/bin/env python3
"""Benchmark the fast diagram core against the five spec targets.

Parses a corpus ONCE with the rust Bobcat backend, then times each
operation the ``lambeq.backend.fast`` core targets, old (stock
grammar/tensor/PytorchModel path) vs fast, over every parsed diagram.
Reports ms/diagram old vs fast vs the spec target and runs the
full-corpus round-trip gate ``to_grammar(to_fast(d)) == d``.

Operations (see the 2026-06-12-fast-diagram-core-design.md spec):

    construction:  tree.to_diagram()      vs tree.to_fast_diagram()
    remove_cups:   RemoveCupsRewriter()(d) vs normal.remove_snakes(fd)
    ansatz:        SpiderAnsatz functor    vs FFunctor (proxy)
    copy:          fast_deepcopy(d)        vs fd (shared immutable ref)
    hash/eq:       hash(d), d == d2        vs hash(fd), fd == fd2
    substitution:  deepcopy+mutate / step  vs to_contraction + evaluate

Honest labels:
  * The "ansatz" fast side is a structural-functor-throughput proxy:
    a real ``FFunctor`` mapping every atom to ``Dim(4)`` and building
    a ``Symbol``-carrying image box per word/box.  It reproduces the
    functor traversal + per-box image cost the benchmark targets, but
    NOT SpiderAnsatz's box-splitting (max_order) maths.  The numeric
    equivalence of the contraction path is gated separately by
    tests/backend/test_fast_contraction.py.
  * "remove_cups" pairs the stock ``RemoveCupsRewriter`` (old) with
    ``normal.remove_snakes`` (fast) per the spec's operation list.
    These are related-but-distinct rewriting passes (RemoveCups bends
    cups into caps + transposed boxes; remove_snakes yanks cup/cap
    snake pairs), so this row is a rewriting-cost-class comparison,
    not an equivalence.  The gated equivalence
    (tests/backend/test_fast_normal.py) is remove_snakes ==
    grammar.Diagram.remove_snakes; that fair same-transformation
    comparison is printed as an extra row.

Example:
    python benchmarks/fastdiag_bench.py /tmp/coco_bench.txt
    python benchmarks/fastdiag_bench.py /tmp/coco_bench.txt --num 1000
"""
import argparse
import sys
import time

import torch

from lambeq import (AtomicType, BobcatParser, PytorchModel,
                    RemoveCupsRewriter, SpiderAnsatz, VerbosityLevel)
from lambeq.backend.fast import contraction, convert, FBox, FFunctor, FTy
from lambeq.backend.fast.convert import register_dim
from lambeq.backend.fast.model import FastPytorchModel
from lambeq.backend.fast.normal import remove_snakes
from lambeq.backend.symbol import Symbol
from lambeq.backend.tensor import Dim
from lambeq.core.utils import fast_deepcopy


def best_of(fn, repeats):
    """Best (min) wall time of ``fn`` over ``repeats``, warm once."""
    fn()                                       # warm up (JIT/cache/alloc)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def per_diagram_ms(seconds, n):
    return seconds / n * 1000 if n else float('nan')


# ---------------------------------------------------------------------------
# Ansatz proxy: a real FFunctor mapping every atom -> Dim(4) and every
# word/box -> a Symbol-carrying image box.  See module docstring.
# ---------------------------------------------------------------------------
_DIM4 = register_dim(4)


def _proxy_ob(_functor, _atom_id):
    return FTy((_DIM4,))


def _proxy_ar(functor, box):
    sym = Symbol(box.name,
                 directed_dom=4 ** len(box.dom),
                 directed_cod=4 ** len(box.cod))
    return FBox(box.name, functor(box.dom), functor(box.cod),
                box.kind, box.z, box.is_dagger, sym)


def _spider_ansatz():
    return SpiderAnsatz({t: Dim(4) for t in (
        AtomicType.NOUN, AtomicType.SENTENCE, AtomicType.NOUN_PHRASE,
        AtomicType.PREPOSITIONAL_PHRASE, AtomicType.CONJUNCTION,
        AtomicType.PUNCTUATION)})


def main():
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('sentence_file',
                      help='text file with one sentence per line')
    argp.add_argument('--num', type=int, default=None,
                      help='use only the first NUM sentences')
    argp.add_argument('--repeats', type=int, default=3,
                      help='timed repeats per op (best is reported)')
    argp.add_argument('--device', default='cpu',
                      help='torch device for the substitution weights')
    args = argp.parse_args()

    with open(args.sentence_file) as f:
        sentences = [line.split() for line in f if line.strip()]
    sentences = sentences[:args.num]

    parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                          parser_backend='rust')
    trees = parser.sentences2trees(sentences, tokenised=True,
                                   suppress_exceptions=True)
    trees = [t for t in trees if t is not None]
    n = len(trees)
    print(f'command:          {" ".join(sys.argv[1:])}')
    print(f'parsed:           {n}/{len(sentences)} sentences '
          f'(rust backend)')

    # Precompute the structures each op consumes (NOT timed here).
    grammar_diagrams = [t.to_diagram() for t in trees]
    fast_diagrams = [convert.to_fast(d) for d in grammar_diagrams]
    fast_diagrams2 = [convert.to_fast(d) for d in grammar_diagrams]
    grammar_copies = fast_deepcopy(grammar_diagrams)

    # ---- round-trip gate over ALL diagrams --------------------------
    bad = [i for i, d in enumerate(grammar_diagrams)
           if convert.to_grammar(fast_diagrams[i]) != d]
    print(f'round-trip:       {n - len(bad)}/{n} identical')
    if bad:
        for i in bad[:5]:
            print(f'  MISMATCH at diagram {i}: {grammar_diagrams[i]}')

    # ---- ansatz inputs: remove_cups'd diagrams that SpiderAnsatz can
    #      handle (same set used for old and fast, fair comparison) ----
    remove_cups = RemoveCupsRewriter()
    probe = _spider_ansatz()
    ok_rc = []
    ok_frc = []
    for d in grammar_diagrams:
        try:
            rc = remove_cups(d)
            probe(rc)
        except Exception:
            continue
        ok_rc.append(rc)
        ok_frc.append(convert.to_fast(rc))
    n_ans = len(ok_rc)

    # ---- substitution inputs: SpiderAnsatz circuits + a PytorchModel
    #      for real symbols/weights; keep only circuits both paths can
    #      contract (fair set) -----------------------------------------
    ansatz = _spider_ansatz()
    circuits = [ansatz(rc) for rc in ok_rc]
    model = PytorchModel.from_diagrams(circuits)
    torch.manual_seed(0)
    model.initialise_weights()
    if args.device != 'cpu':
        model = model.to(args.device)
    parameters = dict(zip(model.symbols, model.weights))

    sub_circuits = []
    fast_specs = []
    for c in circuits:
        try:
            spec = contraction.to_contraction(convert.to_fast(c))
            contraction.evaluate(spec, parameters)
        except Exception:
            continue
        sub_circuits.append(c)
        fast_specs.append(spec)
    n_sub = len(sub_circuits)

    # FastPytorchModel sharing the exact same weights, for the true
    # full per-step comparison (old deepcopy+mutate+tn-contract vs fast
    # gather+einsum).
    fast_model = FastPytorchModel.from_diagrams(sub_circuits)
    fast_model.symbols = model.symbols
    fast_model.weights = model.weights

    # ---- timed closures --------------------------------------------
    def old_construction():
        return [t.to_diagram() for t in trees]

    def fast_construction():
        return [t.to_fast_diagram() for t in trees]

    def old_remove_cups():
        rw = RemoveCupsRewriter()
        return [rw(d) for d in grammar_diagrams]

    def old_remove_snakes():
        return [d.remove_snakes() for d in grammar_diagrams]

    def fast_remove_snakes():
        return [remove_snakes(fd) for fd in fast_diagrams]

    def old_ansatz():
        ans = _spider_ansatz()
        return [ans(rc) for rc in ok_rc]

    def fast_ansatz():
        ff = FFunctor(ob=_proxy_ob, ar=_proxy_ar)
        return [ff(frc) for frc in ok_frc]

    def old_copy():
        return fast_deepcopy(grammar_diagrams)

    def old_hash_eq():
        return [(hash(d), d == c)
                for d, c in zip(grammar_diagrams, grammar_copies)]

    def fast_hash_eq():
        return [(hash(fd), fd == fd2)
                for fd, fd2 in zip(fast_diagrams, fast_diagrams2)]

    def old_substitution():
        copies = fast_deepcopy(sub_circuits)
        for diagram in copies:
            for b in diagram.boxes:
                if isinstance(b.data, Symbol):
                    b.data = parameters[b.data]
        return copies

    def fast_extraction():
        return [contraction.to_contraction(convert.to_fast(c))
                for c in sub_circuits]

    def fast_substitution():
        return [contraction.evaluate(spec, parameters)
                for spec in fast_specs]

    # Contract per diagram (corpus circuits have mixed output shapes, so
    # a single stacked call would fail for both models alike).
    def old_full_step():
        return [model.get_diagram_output([c]) for c in sub_circuits]

    def fast_full_step():
        return [fast_model.get_diagram_output([c]) for c in sub_circuits]

    r = args.repeats
    rows = []   # (name, old_ms, fast_ms, target, count)

    o = per_diagram_ms(best_of(old_construction, r), n)
    fst = per_diagram_ms(best_of(fast_construction, r), n)
    rows.append(('construction', o, fst, '<=0.3', n))

    o = per_diagram_ms(best_of(old_remove_cups, r), n)
    fst = per_diagram_ms(best_of(fast_remove_snakes, r), n)
    rows.append(('remove_cups*', o, fst, '<=0.4', n))

    o = per_diagram_ms(best_of(old_ansatz, r), n_ans)
    fst = per_diagram_ms(best_of(fast_ansatz, r), n_ans)
    rows.append(('ansatz(proxy)', o, fst, '<=0.75', n_ans))

    o = per_diagram_ms(best_of(old_copy, r), n)
    rows.append(('copy', o, 0.0, '~0', n))

    o = per_diagram_ms(best_of(old_hash_eq, r), n)
    fst = per_diagram_ms(best_of(fast_hash_eq, r), n)
    rows.append(('hash/eq', o, fst, '-', n))

    o = per_diagram_ms(best_of(old_substitution, r), n_sub)
    fst = per_diagram_ms(best_of(fast_substitution, r), n_sub)
    rows.append(('substitution', o, fst, '-', n_sub))

    extract_ms = per_diagram_ms(best_of(fast_extraction, r), n_sub)

    # true full per-step comparison (both include the contraction)
    o_step = per_diagram_ms(best_of(old_full_step, r), n_sub)
    f_step = per_diagram_ms(best_of(fast_full_step, r), n_sub)

    # fair, gated same-transformation comparison
    o_rs = per_diagram_ms(best_of(old_remove_snakes, r), n)
    f_rs = per_diagram_ms(best_of(fast_remove_snakes, r), n)

    # ---- table ------------------------------------------------------
    print()
    print(f'{"operation":<16}{"old ms":>10}{"fast ms":>10}'
          f'{"ratio":>9}{"target":>9}{"n":>7}')
    print('-' * 61)
    for name, old_ms, fast_ms, target, count in rows:
        if fast_ms and fast_ms == fast_ms:        # not nan, not zero
            ratio = f'{old_ms / fast_ms:.0f}x'
        elif fast_ms == 0.0:
            ratio = 'inf'
        else:
            ratio = '-'
        fast_disp = f'{fast_ms:.4f}' if fast_ms else '0.0000'
        print(f'{name:<16}{old_ms:>10.4f}{fast_disp:>10}'
              f'{ratio:>9}{target:>9}{count:>7}')
    print('-' * 61)
    print('* remove_cups: old=RemoveCupsRewriter, fast=remove_snakes '
          '(related but distinct passes; see note below).')
    print(f'  fair same-transform row  -- '
          f'grammar.remove_snakes {o_rs:.4f} ms  vs  '
          f'fast.remove_snakes {f_rs:.4f} ms  '
          f'({o_rs / f_rs:.0f}x)')
    print(f'  substitution extraction (one-off): {extract_ms:.4f} ms/diagram '
          f'(fast amortizes this; old pays deepcopy EVERY step)')
    print(f'  full per-step get_diagram_output -- '
          f'PytorchModel {o_step:.4f} ms  vs  '
          f'FastPytorchModel {f_step:.4f} ms  '
          f'({o_step / f_step:.1f}x)  [both include the contraction]')
    print()
    print('Notes: ansatz fast side is a structural-functor-throughput '
          'proxy (every atom -> Dim(4), Symbol-bearing image box); it is '
          'NOT the literal SpiderAnsatz box-split. copy fast side is a '
          'shared immutable reference (nothing to copy). substitution old '
          'side is deepcopy+symbol-mutate only (it must still contract '
          'afterward), so the row understates the fast win.')

    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
