#!/usr/bin/env python3
"""Prepare wikitext-103 sentences for Bobcat distillation.

Usage: python tools/distill/prepare_data.py OUT_DIR [--max-train N]
Writes train.jsonl, heldout.jsonl (2000), dev.jsonl (64) — one JSON
list of tokens per line.
"""
import argparse
import json
import random
import re
from pathlib import Path

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')
TOKEN_OK = re.compile(r'^[\x20-\x7e]+$')  # printable ASCII only


def sentences_from_wikitext():
    from datasets import load_dataset
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1',
                      split='train')
    for record in ds:
        text = record['text'].strip()
        if not text or text.startswith('='):
            continue
        text = text.replace(' @-@ ', '-').replace(' @.@ ', '.') \
                   .replace(' @,@ ', ',').replace('<unk>', '')
        for sent in SENT_SPLIT.split(text):
            yield sent.strip()


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('out_dir')
    argp.add_argument('--max-train', type=int, default=1_500_000)
    args = argp.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    kept: list[list[str]] = []
    target = args.max_train + 2064
    for sent in sentences_from_wikitext():
        words = sent.split()
        if not 4 <= len(words) <= 60:
            continue
        if not all(TOKEN_OK.match(w) for w in words):
            continue
        key = ' '.join(words)
        if key in seen:
            continue
        seen.add(key)
        kept.append(words)
        if len(kept) >= target:
            break

    random.Random(0).shuffle(kept)
    splits = {'heldout.jsonl': kept[:2000],
              'dev.jsonl': kept[2000:2064],
              'train.jsonl': kept[2064:]}
    for name, rows in splits.items():
        with open(out / name, 'w') as f:
            for words in rows:
                f.write(json.dumps(words) + '\n')
        print(f'{name}: {len(rows)} sentences')


if __name__ == '__main__':
    main()
