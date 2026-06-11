#!/usr/bin/env python3
"""Initialise a depth-pruned Bobcat student from the teacher.

Usage: python tools/distill/make_student.py TEACHER_DIR STUDENT_DIR --layers 6
Copies embeddings, heads and equally spaced encoder layers.
"""
import argparse
import re

import torch
from transformers import AutoTokenizer

from lambeq.bobcat import BertForChartClassification

LAYER_RE = re.compile(r'(bert\.encoder\.layer\.)(\d+)(\..*)')


def layer_map(teacher_layers: int, student_layers: int) -> list[int]:
    if student_layers == 1:
        return [teacher_layers - 1]
    return [round(i * (teacher_layers - 1) / (student_layers - 1))
            for i in range(student_layers)]


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('teacher_dir')
    argp.add_argument('student_dir')
    argp.add_argument('--layers', type=int, default=6)
    args = argp.parse_args()

    teacher = BertForChartClassification.from_pretrained(args.teacher_dir)
    config = teacher.config
    t_layers = config.num_hidden_layers
    mapping = layer_map(t_layers, args.layers)
    print(f'layer map (student -> teacher): {mapping}')

    config.num_hidden_layers = args.layers
    student = BertForChartClassification(config)

    teacher_sd = teacher.state_dict()
    new_sd = {}
    for key in student.state_dict():
        m = LAYER_RE.match(key)
        src = (f'{m.group(1)}{mapping[int(m.group(2))]}{m.group(3)}'
               if m else key)
        new_sd[key] = teacher_sd[src].clone()
    student.load_state_dict(new_sd, strict=True)
    student.save_pretrained(args.student_dir)
    AutoTokenizer.from_pretrained(args.teacher_dir).save_pretrained(
        args.student_dir)
    n = sum(p.numel() for p in student.parameters())
    print(f'student saved: {args.layers} layers, {n/1e6:.0f}M params')


if __name__ == '__main__':
    main()
