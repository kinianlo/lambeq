#!/usr/bin/env python3
"""Benchmark BobcatParser throughput and memory.

Reads a text file with one (whitespace-tokenisable) sentence per line and
reports tagging time, chart-parsing time, sentences/sec and peak memory.

Example:
    python benchmarks/bobcat_throughput.py sentences.txt --device cuda
"""
import argparse
import resource
import time

import torch

from lambeq import BobcatParser, VerbosityLevel


def peak_rss_mb() -> float:
    """Peak resident set size of this process and its children, in MB."""
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (self_kb + children_kb) / 1024


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('sentence_file',
                      help='text file with one sentence per line')
    argp.add_argument('--device', default='cpu')
    argp.add_argument('--num', type=int, default=None,
                      help='use only the first NUM sentences')
    argp.add_argument('--batch-size', type=int, default=None)
    argp.add_argument('--max-spans-per-batch', type=int, default=None)
    argp.add_argument('--dtype', default=None,
                      help="e.g. 'float16' or 'bfloat16'")
    argp.add_argument('--n-jobs', type=int, default=1)
    args = argp.parse_args()

    with open(args.sentence_file) as f:
        sentences = [line.split() for line in f if line.strip()]
    sentences = sentences[:args.num]

    kwargs = {}
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    if args.max_spans_per_batch is not None:
        kwargs['max_spans_per_batch'] = args.max_spans_per_batch
    if args.dtype is not None:
        kwargs['dtype'] = args.dtype

    parser = BobcatParser(device=args.device,
                          verbose=VerbosityLevel.SUPPRESS.value,
                          **kwargs)

    parse_kwargs = {}
    if args.n_jobs != 1:
        parse_kwargs['n_jobs'] = args.n_jobs

    cuda = torch.device(args.device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    # warm-up: initialise lazy state (e.g. CUDA kernels)
    parser.sentences2trees([['Alice', 'likes', 'Bob']], tokenised=True)

    # Stage timing: the tagger is timed standalone, then the full
    # pipeline is timed; chart-parse time is the difference (the tagger
    # therefore runs twice, which is accepted for simplicity).
    start = time.perf_counter()
    parser.tagger(sentences, verbose=VerbosityLevel.SUPPRESS.value)
    if cuda:
        torch.cuda.synchronize()
    tag_time = time.perf_counter() - start

    start = time.perf_counter()
    trees = parser.sentences2trees(sentences,
                                   tokenised=True,
                                   suppress_exceptions=True,
                                   **parse_kwargs)
    pipeline_time = time.perf_counter() - start

    parse_time = pipeline_time - tag_time
    n = len(sentences)
    failures = sum(tree is None for tree in trees)

    print(f'config:           {kwargs} {parse_kwargs} '
          f'device={args.device}')
    print(f'sentences:        {n} ({failures} failed)')
    print(f'tagging:          {tag_time:.2f}s ({n / tag_time:.1f} sent/s)')
    print(f'chart parsing:    {parse_time:.2f}s ({n / parse_time:.1f} sent/s)')
    print(f'end-to-end:       {pipeline_time:.2f}s '
          f'({n / pipeline_time:.1f} sent/s)')
    print(f'peak RSS:         {peak_rss_mb():.0f} MB')
    if cuda:
        print(f'peak CUDA memory: '
              f'{torch.cuda.max_memory_allocated() / 2**20:.0f} MB')


if __name__ == '__main__':
    main()
