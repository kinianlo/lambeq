#!/usr/bin/env python3
"""Benchmark the Rust diagram construction kernel against Python-direct.

Parses a corpus ONCE with the Bobcat rust backend, asserts Rust output
equals Python-direct over the full corpus, then measures:

  python-direct:   [t.to_fast_diagram(backend='python') for t in trees]
  rust-end-to-end: trees_to_fast_diagrams(trees, backend='rust')

  Phase split for the Rust path:
    emit:        Python post-order program emission (_emit_program)
    build-serial: bobcat_rs.build_diagrams(programs, num_threads=1)
    build-rayon:  bobcat_rs.build_diagrams(programs, num_threads=0)
    materialize:  [convert.rs_to_fast(rs) for rs in rsdiagrams]

Reports ms/diagram and diagrams/s for each; computes the end-to-end
speedup and compares the build-phase throughput against the ~1700
sent/s parser throughput to determine whether construction is still
the bottleneck.

Best-of-3 repeats with one warm-up pass.

Example:
    python benchmarks/rust_construct_bench.py /tmp/coco_bench.txt
    python benchmarks/rust_construct_bench.py /tmp/coco_bench.txt \\
        --num 2000
"""
import argparse
import sys
import time

import bobcat_rs

from lambeq import BobcatParser, VerbosityLevel
from lambeq.backend.fast import convert
from lambeq.text2diagram.ccg_tree import _emit_program, trees_to_fast_diagrams

# Throughput of the Bobcat CKY parser (for bottleneck comparison).
_PARSER_SENT_PER_S = 1700.0


def best_of(fn, repeats):
    """Best (min) wall time of ``fn`` over ``repeats``, warm once."""
    fn()                                   # warm up (JIT/cache/alloc)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def per_diagram_ms(seconds, n):
    return seconds / n * 1000 if n else float('nan')


def per_diagram_s(seconds, n):
    return n / seconds if seconds and n else float('nan')


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
    n = len(trees)
    print(f'command:          {" ".join(sys.argv[1:])}')
    print(f'parsed:           {n}/{len(sentences)} sentences '
          f'(rust backend)')

    # ------------------------------------------------------------------
    # Correctness gate: Rust must equal Python-direct over the corpus
    # before any timing numbers are recorded.
    # ------------------------------------------------------------------
    rust_list = trees_to_fast_diagrams(trees, backend='rust')
    py_list = [t.to_fast_diagram(backend='python') for t in trees]
    mismatches = [i for i, (r, p) in enumerate(zip(rust_list, py_list))
                  if r != p]
    ok = n - len(mismatches)
    print(f'rust == python:   {ok}/{n} identical')
    if mismatches:
        for i in mismatches[:5]:
            print(f'  MISMATCH at tree {i}')
        print('ERROR: Rust output does not match Python-direct. '
              'Aborting -- numbers would be from a broken build.')
        sys.exit(1)

    # ------------------------------------------------------------------
    # Pre-emit programs once (the emit closure re-emits from scratch
    # each timing call; we only keep this for build/materialize phases).
    # ------------------------------------------------------------------
    def _make_programs():
        progs = []
        for t in trees:
            prog = []
            _emit_program(t.collapse_noun_phrases()._resolved(), prog)
            progs.append(prog)
        return progs

    programs = _make_programs()        # used for phase measurements
    rsdiagrams_serial = bobcat_rs.build_diagrams(programs, 1)

    # ------------------------------------------------------------------
    # Timed closures
    # ------------------------------------------------------------------
    def fn_python_direct():
        return [t.to_fast_diagram(backend='python') for t in trees]

    def fn_rust_e2e():
        return trees_to_fast_diagrams(trees, backend='rust')

    def fn_emit():
        return _make_programs()

    def fn_build_serial():
        return bobcat_rs.build_diagrams(programs, 1)

    def fn_build_rayon():
        return bobcat_rs.build_diagrams(programs, 0)

    def fn_materialize():
        return [convert.rs_to_fast(rs) for rs in rsdiagrams_serial]

    r = args.repeats

    t_python = best_of(fn_python_direct, r)
    t_rust_e2 = best_of(fn_rust_e2e, r)
    t_emit = best_of(fn_emit, r)
    t_bserial = best_of(fn_build_serial, r)
    t_brayon = best_of(fn_build_rayon, r)
    t_mat = best_of(fn_materialize, r)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    speedup = t_python / t_rust_e2 if t_rust_e2 else float('nan')
    rayon_scale = t_bserial / t_brayon if t_brayon else float('nan')

    build_serial_s = per_diagram_s(t_bserial, n)
    build_rayon_s = per_diagram_s(t_brayon, n)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print(f'repeats (best-of):  {r}')
    print()

    # End-to-end table
    fmt_head = f'{"operation":<24}{"ms/diagram":>12}{"diagrams/s":>12}'
    sep = '-' * len(fmt_head)
    print(fmt_head)
    print(sep)

    def row(name, t_sec):
        ms = per_diagram_ms(t_sec, n)
        dps = per_diagram_s(t_sec, n)
        print(f'{name:<24}{ms:>12.4f}{dps:>12.1f}')

    row('python-direct', t_python)
    row('rust-end-to-end', t_rust_e2)
    print(sep)
    print(f'  end-to-end speedup: {speedup:.2f}x  '
          f'(rust vs python-direct)')
    print()

    # Phase split table
    fmt_head2 = (f'{"phase":<24}{"ms/diagram":>12}{"diagrams/s":>12}'
                 f'{"% of rust e2e":>14}')
    sep2 = '-' * len(fmt_head2)

    def pct(t_sec):
        return t_sec / t_rust_e2 * 100 if t_rust_e2 else float('nan')

    print(fmt_head2)
    print(sep2)

    def row2(name, t_sec):
        ms = per_diagram_ms(t_sec, n)
        dps = per_diagram_s(t_sec, n)
        p = pct(t_sec)
        print(f'{name:<24}{ms:>12.4f}{dps:>12.1f}{p:>13.1f}%')

    row2('emit (Python)', t_emit)
    row2('build-serial (Rust)', t_bserial)
    row2('build-rayon (Rust)', t_brayon)
    row2('materialize (Python)', t_mat)
    print(sep2)
    print(f'  rayon scaling: {rayon_scale:.2f}x  '
          f'(serial / all-cores build)')
    print()

    # Comparison vs parser throughput
    parser_ms = 1000.0 / _PARSER_SENT_PER_S
    print('Build-phase throughput vs parser:')
    print(f'  parser          : {_PARSER_SENT_PER_S:.0f} sent/s  '
          f'({parser_ms:.4f} ms/sent)')
    print(f'  build-serial    : {build_serial_s:.1f} diagrams/s  '
          f'({per_diagram_ms(t_bserial, n):.4f} ms/diagram)')
    print(f'  build-rayon     : {build_rayon_s:.1f} diagrams/s  '
          f'({per_diagram_ms(t_brayon, n):.4f} ms/diagram)')
    build_rayon_faster = build_rayon_s > _PARSER_SENT_PER_S
    if build_rayon_faster:
        bottleneck_msg = 'NO -- build-rayon exceeds parser'
    else:
        bottleneck_msg = 'YES -- build-rayon still below parser'
    print(f'  construction bottleneck: {bottleneck_msg}')

    mat_pct = pct(t_mat)
    amdahl_msg = 'YES' if mat_pct > 50 else 'NO'
    print(f'  materialize is {mat_pct:.1f}% of rust-end-to-end '
          f'(Amdahl limiter: {amdahl_msg})')


if __name__ == '__main__':
    main()
