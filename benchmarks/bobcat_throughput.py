#!/usr/bin/env python3
"""Benchmark BobcatParser throughput and memory.

Reads a text file with one (whitespace-tokenisable) sentence per
line and reports tagging time, chart-parsing time, sentences/sec
and peak memory.
The input is assumed to be clean: non-empty,
whitespace-tokenisable lines.

Example:
    python benchmarks/bobcat_throughput.py sentences.txt --device cuda
"""
import argparse
import resource
import sys
import time

import torch

from lambeq import BobcatParser, VerbosityLevel


def peak_rss_mb() -> float:
    """Peak resident set size of process and children, in MB."""
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (self_kb + children_kb) / 1024


def rate(n: int, seconds: float) -> str:
    return f'{n / seconds:.1f} sent/s' if seconds > 0 else 'n/a'


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('sentence_file',
                      help='text file with one sentence per line')
    argp.add_argument('--device', default='cpu')
    argp.add_argument('--num', type=int, default=None,
                      help='use only the first NUM sentences')
    # --batch-size works today; the remaining flags are wired for
    # features added in later tasks of the throughput plan and fail
    # with a TypeError until those tasks land
    argp.add_argument('--batch-size', type=int, default=None)
    argp.add_argument('--max-spans-per-batch', type=int, default=None)
    argp.add_argument('--dtype', default=None,
                      help="e.g. 'float16' or 'bfloat16'")
    argp.add_argument('--n-jobs', type=int, default=1)
    argp.add_argument('--parser-backend', default=None)
    argp.add_argument('--model-dir', default=None)
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
    if args.parser_backend is not None:
        kwargs['parser_backend'] = args.parser_backend

    parser = BobcatParser(model_name_or_path=args.model_dir or 'bobcat',
                          device=args.device,
                          verbose=VerbosityLevel.SUPPRESS.value,
                          **kwargs)

    cuda = torch.device(args.device).type == 'cuda'

    # warm-up: initialise lazy state (e.g. CUDA kernels)
    parser.sentences2trees([['Alice', 'likes', 'Bob']], tokenised=True,
                           suppress_exceptions=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    # stage 1: supertagging
    start = time.perf_counter()
    tag_results = parser.tagger(sentences,
                                verbose=VerbosityLevel.SUPPRESS.value)
    if cuda:
        torch.cuda.synchronize()
    tag_time = time.perf_counter() - start

    # stage 2: serial chart parsing of the tagged sentences, timed
    # directly by mirroring the loop in BobcatParser.sentences2trees
    failures = 0
    start = time.perf_counter()
    for sent in tag_results.sentences:
        try:
            sentence_input = parser._prepare_sentence(sent, tag_results.tags)
            result = parser.parser(sentence_input)
            parser._build_ccgtree(result[0])
        except Exception:
            failures += 1
    parse_time = time.perf_counter() - start

    n = len(sentences)
    total = tag_time + parse_time
    print(f'command:          {" ".join(sys.argv[1:])}')
    # getattr guards: this script is also run against older checkouts
    # of lambeq/ (per-tier comparisons) that predate these attributes
    print(f'tagger: batch_size={parser.tagger.batch_size} '
          f"dtype={getattr(parser.tagger, 'dtype', None)}")
    print(f'config:           {kwargs} device={args.device}')
    print(f'sentences:        {n} ({failures} failed)')
    print(f'tagging:          {tag_time:.2f}s ({rate(n, tag_time)})')
    print(f'chart parsing:    {parse_time:.2f}s ({rate(n, parse_time)})')

    if getattr(parser, 'parser_backend', 'python') == 'rust':
        sentence_inputs = [parser._prepare_sentence(s, tag_results.tags)
                           for s in tag_results.sentences]
        start = time.perf_counter()
        parser.parser.parse_batch(sentence_inputs)
        batch_time = time.perf_counter() - start
        print(f'chart parse_batch: {batch_time:.2f}s '
              f'({rate(n, batch_time)})')
        start = time.perf_counter()
        parser.sentences2trees(sentences,
                               tokenised=True,
                               suppress_exceptions=True,
                               verbose=VerbosityLevel.SUPPRESS.value)
        fused_time = time.perf_counter() - start
        print(f'fused end-to-end:  {fused_time:.2f}s ({rate(n, fused_time)})')

    print(f'end-to-end:       {total:.2f}s ({rate(n, total)})')

    if args.n_jobs != 1:
        # parallel chart parsing happens inside sentences2trees;
        # compare its end-to-end time against tag_time + parse_time
        start = time.perf_counter()
        parser.sentences2trees(sentences,
                               tokenised=True,
                               suppress_exceptions=True,
                               n_jobs=args.n_jobs)
        njobs_time = time.perf_counter() - start
        print(f'end-to-end (n_jobs={args.n_jobs}): '
              f'{njobs_time:.2f}s ({rate(n, njobs_time)})')

    print(f'peak RSS:         {peak_rss_mb():.0f} MB')
    if cuda:
        print(f'peak CUDA memory: '
              f'{torch.cuda.max_memory_allocated() / 2**20:.0f} MB')


if __name__ == '__main__':
    main()
