#!/usr/bin/env python3
"""Train PytorchModel vs FastPytorchModel on COCO captions and compare.

Builds circuits from COCO captions (deterministic synthetic labels),
starts both models from identical cloned weights, runs a minimal
full-batch Adam loop on each, and reports the loss-trajectory match and
the per-epoch wall-clock speedup. Exits nonzero if the trajectories
diverge (so the script doubles as a correctness self-check).

Example:
    python benchmarks/fasttrain_bench.py /tmp/coco_bench.txt --num 200
"""
import argparse
import sys
import time
from collections import Counter

import torch

from lambeq import (AtomicType, BobcatParser, PytorchModel,
                    RemoveCupsRewriter, SpiderAnsatz, VerbosityLevel)
from lambeq.backend.fast.model import FastPytorchModel
from lambeq.backend.tensor import Dim

KEYWORDS = {'man', 'woman', 'men', 'women', 'people', 'person', 'boy',
            'girl', 'child', 'dog', 'cat', 'horse', 'bird', 'animal'}


def label_for(tokens):
    return int(any(t.lower() in KEYWORDS for t in tokens))


def build_circuits(token_lists, dim=2):
    parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                          parser_backend='rust')
    trees = parser.sentences2trees(token_lists, tokenised=True,
                                   suppress_exceptions=True)
    ansatz = SpiderAnsatz({t: Dim(dim) for t in AtomicType})
    remove_cups = RemoveCupsRewriter()
    circuits, labels = [], []
    for toks, tree in zip(token_lists, trees):
        if tree is None:
            continue
        circuits.append(ansatz(remove_cups(tree.to_diagram())))
        labels.append(label_for(toks))
    shapes = [tuple(c.cod.dim) for c in circuits]
    modal = Counter(shapes).most_common(1)[0][0]
    keep = [i for i, s in enumerate(shapes) if s == modal]
    return ([circuits[i] for i in keep],
            [labels[i] for i in keep], modal, len(token_lists))


def train(model, circuits, onehot, epochs, lr):
    """Returns (losses, per_epoch_seconds). Warms up before timing."""
    opt = torch.optim.Adam(model.weights, lr=lr)
    model.get_diagram_output(circuits)          # warm up / build spec cache
    losses, times = [], []
    for _ in range(epochs):
        start = time.perf_counter()
        opt.zero_grad()
        out = model.get_diagram_output(circuits)
        loss = ((out - onehot) ** 2).mean()
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - start)
        losses.append(loss.item())
    return losses, times


def main():
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('corpus')
    argp.add_argument('--num', type=int, default=200,
                      help='cap on captions read from the corpus')
    argp.add_argument('--epochs', type=int, default=10)
    argp.add_argument('--lr', type=float, default=0.05)
    argp.add_argument('--seed', type=int, default=0)
    args = argp.parse_args()

    with open(args.corpus) as f:
        token_lists = [line.split() for line in f if line.strip()]
    token_lists = token_lists[:args.num]

    circuits, labels, shape, n_read = build_circuits(token_lists)
    print(f'read {n_read} captions -> {len(circuits)} circuits kept '
          f'(uniform output shape {shape}); '
          f'{sum(labels)} positive / {len(labels) - sum(labels)} negative')
    if len(circuits) < 4:
        print('too few circuits to train', file=sys.stderr)
        sys.exit(2)

    onehot = torch.nn.functional.one_hot(
        torch.tensor(labels), num_classes=2).float()

    old = PytorchModel.from_diagrams(circuits)
    torch.manual_seed(args.seed)
    old.initialise_weights()
    fast = FastPytorchModel.from_diagrams(circuits)
    fast.weights = torch.nn.ParameterList(
        [torch.nn.Parameter(w.detach().clone()) for w in old.weights])

    old_losses, old_times = train(old, circuits, onehot,
                                  args.epochs, args.lr)
    fast_losses, fast_times = train(fast, circuits, onehot,
                                    args.epochs, args.lr)

    maxdiff = max(abs(a - b) for a, b in zip(old_losses, fast_losses))
    print('\nepoch |    old loss |   fast loss')
    for i, (a, b) in enumerate(zip(old_losses, fast_losses)):
        print(f'{i:5d} | {a:11.6f} | {b:11.6f}')
    print(f'max abs trajectory diff: {maxdiff:.2e}')

    old_ms = sum(old_times) / len(old_times) * 1000
    fast_ms = sum(fast_times) / len(fast_times) * 1000
    print('\nper-epoch wall-clock (forward+backward+step):')
    print(f'  PytorchModel     : {old_ms:8.1f} ms/epoch '
          f'({sum(old_times):.2f} s total)')
    print(f'  FastPytorchModel : {fast_ms:8.1f} ms/epoch '
          f'({sum(fast_times):.2f} s total)')
    print(f'  speedup          : {old_ms / fast_ms:.2f}x')

    if maxdiff > 1e-3:
        print(f'\nTRAJECTORY DIVERGENCE: maxdiff {maxdiff:.2e} > 1e-3',
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
