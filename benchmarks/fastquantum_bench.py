#!/usr/bin/env python3
"""Benchmark the fast quantum front-end against the legacy path.

Parses a corpus ONCE with the rust Bobcat backend, then times the
QUANTUM FRONT-END ONLY (cup-reduced grammar.Diagram production) in two
variants over all parsed trees:

    Legacy::

        [RemoveCupsRewriter()(t.to_diagram()) for t in trees]

    Fast::

        compile_quantum_input(trees)
            # to_fast_diagram + fast.remove_cups + convert.to_grammar
            # All on the fast core; output is grammar.Diagram.

Both produce byte-identical grammar.Diagrams suitable for any
CircuitAnsatz (IQPAnsatz, TketAnsatz, PennyLaneAnsatz, ...).  The
quantum ansatz + circuit execution on pennylane/tket are OUT OF SCOPE
and not measured here — this benchmark isolates the diagram-building
speedup only.

Before timing, the script asserts front-end identity on a small slice::

    compile_quantum_input(trees[:k])[i] ==
    RemoveCupsRewriter()(trees[i].to_diagram())

Exits nonzero if any mismatch, so recorded numbers only come from a
correct build.

Example::

    python benchmarks/fastquantum_bench.py /tmp/coco_bench.txt
    python benchmarks/fastquantum_bench.py /tmp/coco_bench.txt --num 400
"""
import argparse
import sys
import time

from lambeq import BobcatParser, RemoveCupsRewriter, VerbosityLevel
from lambeq.backend.fast import compile_quantum_input


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
    argp.add_argument('--num', type=int, default=400,
                      help='use only the first NUM sentences (default 400)')
    argp.add_argument('--repeats', type=int, default=3,
                      help='timed repeats per op (best is reported)')
    args = argp.parse_args()

    with open(args.sentence_file) as f:
        sentences = [line.split() for line in f if line.strip()]
    sentences = sentences[:args.num]

    # ---- parse (rust backend; not timed) ----------------------------
    parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                          parser_backend='rust')
    trees = parser.sentences2trees(sentences, tokenised=True,
                                   suppress_exceptions=True)
    trees = [t for t in trees if t is not None]
    n = len(trees)

    print(f'command:          {" ".join(sys.argv[1:])}')
    print(f'parsed:           {n}/{len(sentences)} sentences '
          f'(rust backend)')

    # ---- front-end identity gate (small slice before timing) --------
    # Assert compile_quantum_input(trees[:k])[i] ==
    #        RemoveCupsRewriter()(trees[i].to_diagram())
    gate_n = min(10, n)
    rc = RemoveCupsRewriter()
    fast_gate = compile_quantum_input(trees[:gate_n])
    mismatches = 0
    for i in range(gate_n):
        expected = rc(trees[i].to_diagram())
        if fast_gate[i] != expected:
            print(f'  IDENTITY MISMATCH at tree {i}: {trees[i]}',
                  file=sys.stderr)
            mismatches += 1
    if mismatches:
        print(f'identity gate FAILED: {mismatches}/{gate_n} mismatches',
              file=sys.stderr)
        sys.exit(1)
    print(f'identity gate:    PASS ({gate_n}/{gate_n} byte-identical)')

    # ---- timed closures ---------------------------------------------
    r = args.repeats

    # Legacy front-end: to_diagram + RemoveCupsRewriter
    def legacy_frontend():
        rw = RemoveCupsRewriter()
        return [rw(t.to_diagram()) for t in trees]

    # Fast front-end: compile_quantum_input (to_fast_diagram +
    # fast.remove_cups + convert.to_grammar — all on the fast core)
    def fast_frontend():
        return compile_quantum_input(trees)

    # ---- time -------------------------------------------------------
    leg_s = best_of(legacy_frontend, r)
    fst_s = best_of(fast_frontend, r)

    leg_ms = per_diagram_ms(leg_s, n)
    fst_ms = per_diagram_ms(fst_s, n)
    speedup = leg_ms / fst_ms if fst_ms else float('inf')

    # ---- results table ----------------------------------------------
    print()
    print(f'{"stage":<32}{"legacy ms":>12}{"fast ms":>10}{"speedup":>10}'
          f'{"n":>7}')
    print('-' * 71)
    print(f'{"quantum front-end":<32}{leg_ms:>12.4f}{fst_ms:>10.4f}'
          f'{speedup:>9.1f}x{n:>7}')
    print('-' * 71)
    print()
    print(f'repeats: {r} (best of {r}, 1 warm-up); '
          f'parser: rust Bobcat backend')
    print()
    print('Notes:')
    print('  legacy  = [RemoveCupsRewriter()(t.to_diagram()) for t in trees]')
    print('  fast    = compile_quantum_input(trees)')
    print('            (to_fast_diagram + fast.remove_cups'
          ' + convert.to_grammar,')
    print('             all on the fast core; output is grammar.Diagram)')
    print('  SCOPE: diagram-building ONLY.  The quantum ansatz + circuit')
    print('  execution on pennylane/tket are unchanged and NOT measured.')
    print('  Circuit execution dominates QML wall-clock; this speedup')
    print('  applies to the preprocessing (dataset-build) phase only.')


if __name__ == '__main__':
    main()
