#!/usr/bin/env python3
"""Verify Rust and Python Bobcat backends produce identical trees.

Usage: python benchmarks/check_equivalence.py sentences.txt [more.txt ...]
Exits non-zero and prints each mismatching sentence.
"""
import sys

from lambeq import BobcatParser, VerbosityLevel


def main() -> int:
    sentences = []
    for path in sys.argv[1:]:
        with open(path) as f:
            sentences += [line.split() for line in f if line.strip()]

    py = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                      parser_backend='python')
    rs = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                      parser_backend='rust')
    # share the tagger weights: drop the second BERT copy
    rs.tagger = py.tagger

    py_trees = py.sentences2trees(sentences, tokenised=True,
                                  suppress_exceptions=True)
    rs_trees = rs.sentences2trees(sentences, tokenised=True,
                                  suppress_exceptions=True)
    bad = 0
    for words, a, b in zip(sentences, py_trees, rs_trees):
        if a != b:
            bad += 1
            print('MISMATCH:', ' '.join(words))
    total = len(sentences)
    print(f'{total - bad}/{total} identical')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
