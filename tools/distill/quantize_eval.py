#!/usr/bin/env python3
"""Dynamic-int8 quantization experiment for the Bobcat tagger (CPU).

Quantizes the Linear layers of a Bobcat model with torch dynamic
quantization, then reports CPU throughput and tree agreement against
the unquantized model.

Usage: python tools/distill/quantize_eval.py MODEL_DIR CORPUS.txt \
    [--agree-jsonl HELDOUT.jsonl] [--agree-limit 300] [--num N]
"""
import argparse
import json
import time

import torch

from lambeq import BobcatParser, VerbosityLevel


def fused_time(parser: BobcatParser, sentences: list[list[str]]) -> float:
    parser.sentences2trees(sentences[:8], tokenised=True,
                           suppress_exceptions=True)  # warm
    start = time.perf_counter()
    parser.sentences2trees(sentences, tokenised=True,
                           suppress_exceptions=True)
    return time.perf_counter() - start


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('model_dir')
    argp.add_argument('corpus')
    argp.add_argument('--agree-jsonl', default=None)
    argp.add_argument('--agree-limit', type=int, default=300)
    argp.add_argument('--num', type=int, default=None)
    args = argp.parse_args()

    with open(args.corpus) as f:
        sentences = [line.split() for line in f if line.strip()]
    sentences = sentences[:args.num]

    common = dict(verbose=VerbosityLevel.SUPPRESS.value, device='cpu',
                  parser_backend='rust')
    fp32 = BobcatParser(model_name_or_path=args.model_dir, **common)
    int8 = BobcatParser(model_name_or_path=args.model_dir, **common)
    int8.tagger.model = torch.ao.quantization.quantize_dynamic(
        int8.tagger.model, {torch.nn.Linear}, dtype=torch.qint8).eval()

    n = len(sentences)
    t_fp32 = fused_time(fp32, sentences)
    t_int8 = fused_time(int8, sentences)
    result = {'corpus': args.corpus, 'sentences': n,
              'fp32_s': round(t_fp32, 2),
              'fp32_sent_s': round(n / t_fp32, 1),
              'int8_s': round(t_int8, 2),
              'int8_sent_s': round(n / t_int8, 1),
              'speedup': round(t_fp32 / t_int8, 2)}

    if args.agree_jsonl:
        with open(args.agree_jsonl) as f:
            agree_sents = [json.loads(line)
                           for line in f][:args.agree_limit]
        a = fp32.sentences2trees(agree_sents, tokenised=True,
                                 suppress_exceptions=True)
        b = int8.sentences2trees(agree_sents, tokenised=True,
                                 suppress_exceptions=True)
        m = len(agree_sents)
        same = sum(1 for x, y in zip(a, b) if x is not None and x == y)
        both_fail = sum(1 for x, y in zip(a, b)
                        if x is None and y is None)
        result.update({'agree_sentences': m,
                       'agreement': round((same + both_fail) / m, 4),
                       'fp32_fail': sum(t is None for t in a),
                       'int8_fail': sum(t is None for t in b)})

    print(json.dumps(result))


if __name__ == '__main__':
    main()
