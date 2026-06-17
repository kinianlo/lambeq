#!/usr/bin/env python3
"""Benchmark FRewriter against the legacy box-level Rewriter.

Parses a corpus ONCE with the rust Bobcat backend, then times:

    legacy:  Rewriter(rules)(d)     per grammar.Diagram.
    convert: convert.to_fast(d)     per diagram (one-off preprocessing;
             reported separately from the rewrite proper).
    fast:    FRewriter(rules)(fd)   per pre-converted fast diagram.

Before timing the script asserts the structural differential on every
diagram: ``to_grammar(FRewriter(r)(to_fast(d))) == Rewriter(r)(d)``. It
also reports how many diagrams the rules actually rewrite (a speedup on
a corpus where no rule fires would be a pure-traversal number only), so
the result is never recorded from a vacuous or broken run.

Example::

    python benchmarks/fastrewriter_bench.py /tmp/coco_bench.txt
    python benchmarks/fastrewriter_bench.py /tmp/coco_bench.txt --num 5
"""
import argparse
import sys
import time

from lambeq import BobcatParser, VerbosityLevel
from lambeq.backend.fast import convert, FRewriter
from lambeq.rewrite import Rewriter

# Default + the extra rules that fire on COCO; mirrors the test's set.
RULES = ['determiner', 'connector', 'coordination', 'curry',
         'prepositional_phrase', 'auxiliary',
         'object_rel_pronoun', 'subject_rel_pronoun']


def best_of(fn, repeats):
    """Best (min) wall time of ``fn`` over ``repeats``, warm once."""
    fn()                                       # warm up (cache/alloc)
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
    n = len(diagrams)

    print(f'command:          {" ".join(sys.argv[1:])}')
    print(f'parsed:           {len(trees)}/{len(sentences)} sentences '
          f'(rust backend)')
    print(f'diagrams:         {n}')

    rw = Rewriter(RULES)
    fr = FRewriter(RULES)

    # ---- structural differential gate + coverage --------------------
    fasts = [convert.to_fast(d) for d in diagrams]
    changed = 0
    mismatches = 0
    for d, fd in zip(diagrams, fasts):
        legacy = rw(d)
        fast = convert.to_grammar(fr(fd))
        if fast != legacy:
            mismatches += 1
        if legacy != d:
            changed += 1
    if mismatches:
        print(f'differential FAILED: {mismatches}/{n} mismatches',
              file=sys.stderr)
        sys.exit(1)
    print(f'differential:     PASS ({n}/{n} structurally identical)')
    print(f'rewritten:        {changed}/{n} diagrams changed by the rules')

    r = args.repeats

    # ---- convert (one-off preprocessing) ----------------------------
    def _do_convert():
        return [convert.to_fast(d) for d in diagrams]

    convert_s = best_of(_do_convert, r)
    convert_ms = per_diagram_ms(convert_s, n)
    fasts = _do_convert()

    # ---- legacy Rewriter timing -------------------------------------
    def _legacy():
        return [rw(d) for d in diagrams]

    legacy_s = best_of(_legacy, r)
    legacy_ms = per_diagram_ms(legacy_s, n)

    # ---- fast FRewriter timing (rewrite proper, not convert) --------
    def _fast():
        return [fr(fd) for fd in fasts]

    fast_s = best_of(_fast, r)
    fast_ms = per_diagram_ms(fast_s, n)

    speedup = legacy_ms / fast_ms if fast_ms else float('inf')

    # ---- results table ----------------------------------------------
    print()
    print(f'{"operation":<22}{"ms/diagram":>12}{"n":>7}')
    print('-' * 41)
    print(f'{"legacy Rewriter":<22}{legacy_ms:>12.4f}{n:>7}')
    print(f'{"fast FRewriter":<22}{fast_ms:>12.4f}{n:>7}')
    print(f'{"convert (one-off)":<22}{convert_ms:>12.4f}{n:>7}')
    print('-' * 41)
    print(f'speedup (legacy/fast):  {speedup:.1f}x')
    print(f'repeats:                {r} (best of {r}, 1 warm-up)')
    print()
    print('Note: a once-per-dataset preprocessing win. Matched-box '
          'fragments still reuse the legacy rule logic, so the speedup '
          'is bounded by how many boxes the rules rewrite (most pass '
          'through as pure FFunctor traversal).')


if __name__ == '__main__':
    main()
