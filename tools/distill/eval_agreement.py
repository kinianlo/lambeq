#!/usr/bin/env python3
"""Tree-agreement evaluation: student Bobcat vs teacher Bobcat.

Usage: python tools/distill/eval_agreement.py TEACHER_DIR STUDENT_DIR \
    SENTENCES.jsonl [--device cuda] [--gate 0.95] [--limit N]
Exit code 1 if --gate is set and agreement falls below it.
"""
import argparse
import json
import sys

from lambeq import BobcatParser, VerbosityLevel


def main() -> int:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('teacher_dir')
    argp.add_argument('student_dir')
    argp.add_argument('sentence_file')
    argp.add_argument('--device', default='cuda')
    argp.add_argument('--gate', type=float, default=None)
    argp.add_argument('--limit', type=int, default=None)
    args = argp.parse_args()

    with open(args.sentence_file) as f:
        sentences = [json.loads(line) for line in f][:args.limit]

    common = dict(verbose=VerbosityLevel.SUPPRESS.value,
                  device=args.device, parser_backend='rust')
    teacher = BobcatParser(model_name_or_path=args.teacher_dir, **common)
    student = BobcatParser(model_name_or_path=args.student_dir, **common)

    t_trees = teacher.sentences2trees(sentences, tokenised=True,
                                      suppress_exceptions=True)
    s_trees = student.sentences2trees(sentences, tokenised=True,
                                      suppress_exceptions=True)

    n = len(sentences)
    same = sum(1 for a, b in zip(t_trees, s_trees)
               if a is not None and a == b)
    both_fail = sum(1 for a, b in zip(t_trees, s_trees)
                    if a is None and b is None)
    t_fail = sum(t is None for t in t_trees)
    s_fail = sum(t is None for t in s_trees)
    agreement = (same + both_fail) / n
    print(json.dumps({'sentences': n,
                      'agreement': round(agreement, 4),
                      'identical_trees': same,
                      'both_fail': both_fail,
                      'teacher_fail': t_fail,
                      'student_fail': s_fail}))
    if args.gate is not None and agreement < args.gate:
        print(f'GATE FAILED: {agreement:.4f} < {args.gate}',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
