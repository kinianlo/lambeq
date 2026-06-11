#!/usr/bin/env python3
"""Assemble a complete drop-in Bobcat model dir from distilled weights.

Usage: python tools/distill/package_student.py STUDENT_WEIGHTS_DIR \
    TEACHER_DIR OUT_DIR
Copies grammar/pipeline/tokenizer/version files from the teacher next
to the student weights.
"""
import argparse
import shutil
from pathlib import Path

COPY = ['grammar.json', 'pipeline_config.json', 'version.txt',
        'tokenizer.json', 'tokenizer_config.json', 'vocab.txt',
        'special_tokens_map.json']


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('student_weights_dir')
    argp.add_argument('teacher_dir')
    argp.add_argument('out_dir')
    args = argp.parse_args()
    src = Path(args.student_weights_dir)
    teacher = Path(args.teacher_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        shutil.copy2(item, out / item.name)
    for name in COPY:
        path = teacher / name
        if path.exists() and not (out / name).exists():
            shutil.copy2(path, out / name)
    print(f'packaged: {out}')


if __name__ == '__main__':
    main()
