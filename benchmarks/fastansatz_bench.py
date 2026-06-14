#!/usr/bin/env python3
"""Benchmark FSpiderAnsatz against the legacy SpiderAnsatz.

Parses a corpus ONCE with the rust Bobcat backend, runs
``RemoveCupsRewriter`` on every diagram, then times:

    legacy:  SpiderAnsatz(ob)(d)   per removed-cups diagram.
    convert: convert.to_fast(d)    per diagram (one-off preprocessing;
             reported separately from the ansatz proper).
    fast:    FSpiderAnsatz(ob)(fd) per pre-converted fast diagram.

``ob`` maps every ``AtomicType`` to ``Dim(2)``.

Before timing the script asserts numeric equivalence on a small
slice: legacy SpiderAnsatz circuit fed through PytorchModel versus
fast FSpiderAnsatz circuit fed through ``contraction.to_contraction``
+ ``contraction.evaluate``, allclose at ``atol=1e-5``.  Exits
nonzero on any mismatch so results from a broken build are never
recorded.

Example::

    python benchmarks/fastansatz_bench.py /tmp/coco_bench.txt
    python benchmarks/fastansatz_bench.py /tmp/coco_bench.txt --num 1000
"""
import argparse
import sys
import time

import torch

from lambeq import (AtomicType, BobcatParser, PytorchModel,
                    RemoveCupsRewriter, SpiderAnsatz, VerbosityLevel)
from lambeq.backend.fast import contraction, convert, FSpiderAnsatz
from lambeq.backend.tensor import Dim


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


def main():
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('sentence_file',
                      help='text file with one sentence per line')
    argp.add_argument('--num', type=int, default=None,
                      help='use only the first NUM sentences')
    argp.add_argument('--repeats', type=int, default=3,
                      help='timed repeats per op (best is reported)')
    args = argp.parse_args()

    with open(args.sentence_file) as f:
        sentences = [line.split() for line in f if line.strip()]
    sentences = sentences[:args.num]

    parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                          parser_backend='rust')
    trees = parser.sentences2trees(sentences, tokenised=True,
                                   suppress_exceptions=True)
    trees = [t for t in trees if t is not None]
    diagrams = [t.to_diagram() for t in trees]

    # Apply RemoveCupsRewriter once — not timed; same set both paths.
    remove_cups = RemoveCupsRewriter()
    nocups = []
    for d in diagrams:
        try:
            nocups.append(remove_cups(d))
        except Exception:
            pass
    n = len(nocups)

    print(f'command:          {" ".join(sys.argv[1:])}')
    print(f'parsed:           {len(trees)}/{len(sentences)} sentences '
          f'(rust backend)')
    print(f'after remove_cups: {n} diagrams')

    ob = {t: Dim(2) for t in AtomicType}

    # ---- numeric equivalence gate (small slice) ---------------------
    # Mirror the Task-2 harness: legacy SpiderAnsatz + PytorchModel vs
    # fast FSpiderAnsatz + to_contraction + evaluate, atol=1e-5.
    gate_n = min(10, n)
    legacy_ansatz = SpiderAnsatz(ob)
    g_circuits = [legacy_ansatz(d) for d in nocups[:gate_n]]
    model = PytorchModel.from_diagrams(g_circuits)
    torch.manual_seed(0)
    model.initialise_weights()
    expected = model.get_diagram_output(g_circuits)
    weights = dict(zip(model.symbols, model.weights))

    fans = FSpiderAnsatz(ob)
    mismatches = 0
    for i, (d, exp) in enumerate(zip(nocups[:gate_n], expected)):
        try:
            fd = fans(convert.to_fast(d))
            spec = contraction.to_contraction(fd)
            got = contraction.evaluate(spec, weights)
            if not torch.allclose(got, exp, atol=1e-5):
                print(f'  NUMERIC MISMATCH at diagram {i}',
                      file=sys.stderr)
                mismatches += 1
        except Exception as e:
            print(f'  ERROR at diagram {i}: {e}', file=sys.stderr)
            mismatches += 1
    if mismatches:
        print(f'numeric gate FAILED: {mismatches}/{gate_n} mismatches',
              file=sys.stderr)
        sys.exit(1)
    print(f'numeric gate:     '
          f'PASS ({gate_n}/{gate_n} allclose at atol=1e-5)')

    # ---- pre-convert fast diagrams (cost reported separately) -------
    # One-off preprocessing: grammar Diagram -> FDiagram.
    # Done outside timing so the ansatz timing is pure ansatz.
    fasts = [convert.to_fast(d) for d in nocups]   # pre-warm
    r = args.repeats

    def _do_convert():
        return [convert.to_fast(d) for d in nocups]

    convert_s = best_of(_do_convert, r)
    convert_ms = per_diagram_ms(convert_s, n)

    # Rebuild fasts from the last warm call.
    fasts = _do_convert()

    # ---- legacy SpiderAnsatz timing ---------------------------------
    def _legacy():
        ans = SpiderAnsatz(ob)
        return [ans(d) for d in nocups]

    legacy_s = best_of(_legacy, r)
    legacy_ms = per_diagram_ms(legacy_s, n)

    # ---- fast FSpiderAnsatz timing (ansatz proper, not convert) -----
    def _fast():
        fans2 = FSpiderAnsatz(ob)
        return [fans2(fd) for fd in fasts]

    fast_s = best_of(_fast, r)
    fast_ms = per_diagram_ms(fast_s, n)

    speedup = legacy_ms / fast_ms if fast_ms else float('inf')

    # ---- results table ----------------------------------------------
    print()
    print(f'{"operation":<22}{"ms/diagram":>12}{"n":>7}')
    print('-' * 41)
    print(f'{"legacy SpiderAnsatz":<22}{legacy_ms:>12.4f}{n:>7}')
    print(f'{"fast FSpiderAnsatz":<22}{fast_ms:>12.4f}{n:>7}')
    print(f'{"convert (one-off)":<22}{convert_ms:>12.4f}{n:>7}')
    print('-' * 41)
    print(f'speedup (legacy/fast):  {speedup:.1f}x')
    print(f'repeats:                {r} (best of {r}, 1 warm-up)')
    print()
    print('Note: "convert" is a once-per-dataset preprocessing cost '
          '(grammar.Diagram -> FDiagram).')
    print('      The training-step bottleneck is ansatz + contraction;'
          ' this benchmark covers ansatz only.')


if __name__ == '__main__':
    main()
