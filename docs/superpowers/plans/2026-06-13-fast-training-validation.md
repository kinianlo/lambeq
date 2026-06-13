# Fast Diagram Core Training Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove `FastPytorchModel` trains identically to the stock `PytorchModel` (same seed/weights/data → matching loss trajectory) and measure the per-epoch wall-clock speedup, on a real COCO→Bobcat→ansatz pipeline.

**Architecture:** A small CI test asserts the two models' loss trajectories stay `allclose` across epochs (the decisive correctness gate), and a standalone benchmark script trains both on N COCO captions with deterministic synthetic labels and reports the wall-clock speedup. Both build the two models from the same circuits, start them from identical cloned weights, and run a minimal full-batch Adam loop. The script and test are independent and self-contained (no shared module, no packaging change).

**Tech Stack:** Python 3.10+, PyTorch, lambeq (`BobcatParser` rust backend, `SpiderAnsatz`, `RemoveCupsRewriter`, `PytorchModel`, `lambeq.backend.fast.model.FastPytorchModel`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-fast-training-validation-design.md`

---

## Conventions (every task)

- Repo `/home/kinianlo/projects/lambeq`, branch `fast-training-validation`.
  `PY=~/.pyenv/versions/qnlp/bin/python`. Use 600000ms timeouts on any
  command that constructs a `BobcatParser` (loads the cached model).
- Corpus: `/tmp/coco_bench.txt` (2000 tokenized COCO captions, already on
  disk). Bobcat model cached at `~/.cache/lambeq/bobcat/bobcat`.
- Commit style: plain sentence + trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- This work is ADDITIVE: it creates a test and a benchmark script and
  appends to RESULTS.md. It does NOT modify any library code under
  `lambeq/`. If a task seems to require a library change, STOP and report
  it — a needed library change is a finding (the fast core not training
  correctly), not a step to paper over.

## Established facts (verified, do not re-derive)

- `FastPytorchModel` is at `lambeq.backend.fast.model.FastPytorchModel`,
  a `PytorchModel` subclass overriding `get_diagram_output`.
- A single-backward gradient check already matched the stock model
  (maxdiff 1.7e-6 over 1328 params), so trajectories MUST match under
  identical init + full-batch updates; a divergence is a real bug.
- `Model.from_diagrams(circuits)` builds a model and populates
  `.symbols`; `.initialise_weights()` fills `.weights` (a
  `torch.nn.ParameterList`) deterministically given a torch seed.
- `AtomicType` iterates to the five canonical types (`n, s, p, conj,
  punc`); `{t: Dim(2) for t in AtomicType}` is a valid `SpiderAnsatz`
  ob_map.
- A tensor circuit's output shape is `tuple(circuit.cod.dim)`; a
  well-formed declarative reduces to `Dim(2)` → shape `(2,)` after
  `RemoveCupsRewriter` + this ansatz.
- `PytorchModel.get_diagram_output(circuits)` returns
  `torch.stack([...])`, so every circuit in one call MUST share an output
  shape — hence the uniform-shape filter.

## File map

```
tests/backend/test_fast_training.py   CI trajectory-match gate (small, no library change)
benchmarks/fasttrain_bench.py         standalone train-both benchmark + self-check
benchmarks/RESULTS.md                 append "Fast diagram core — training" section
```

---

### Task 1: Trajectory-match CI gate (`tests/backend/test_fast_training.py`)

**Files:**
- Create: `tests/backend/test_fast_training.py`

This test is the deliverable; the code it exercises (`FastPytorchModel`)
already exists, so a correctly-written test PASSES. If it FAILS, that is a
real finding about the fast core — report it, do not weaken the assertion.

- [ ] **Step 1: Write the test**

```python
# tests/backend/test_fast_training.py
import pytest

torch = pytest.importorskip('torch')

from lambeq import (AtomicType, BobcatParser, PytorchModel,
                    RemoveCupsRewriter, SpiderAnsatz, VerbosityLevel)
from lambeq.backend.fast.model import FastPytorchModel
from lambeq.backend.tensor import Dim

KEYWORDS = {'man', 'woman', 'men', 'women', 'people', 'person', 'boy',
            'girl', 'child', 'dog', 'cat', 'horse', 'bird', 'animal'}


def _label(tokens):
    return int(any(t.lower() in KEYWORDS for t in tokens))


def _build_circuits(token_lists, dim=2):
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
        labels.append(_label(toks))
    return circuits, labels


def _filter_uniform_shape(circuits, labels):
    from collections import Counter
    shapes = [tuple(c.cod.dim) for c in circuits]
    modal = Counter(shapes).most_common(1)[0][0]
    keep = [i for i, s in enumerate(shapes) if s == modal]
    return ([circuits[i] for i in keep], [labels[i] for i in keep], modal)


def _train(model, circuits, labels, epochs, lr):
    onehot = torch.nn.functional.one_hot(
        torch.tensor(labels), num_classes=2).float()
    opt = torch.optim.Adam(model.weights, lr=lr)
    model.get_diagram_output(circuits)          # warm up / build spec cache
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        out = model.get_diagram_output(circuits)
        loss = ((out - onehot) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_fast_model_training_trajectory_matches():
    with open('/tmp/coco_bench.txt') as f:
        token_lists = [line.split() for line in f if line.strip()][:30]
    circuits, labels = _build_circuits(token_lists)
    circuits, labels, shape = _filter_uniform_shape(circuits, labels)
    circuits, labels = circuits[:12], labels[:12]
    assert len(circuits) >= 4 and shape == (2,)

    old = PytorchModel.from_diagrams(circuits)
    torch.manual_seed(0)
    old.initialise_weights()

    fast = FastPytorchModel.from_diagrams(circuits)
    fast.symbols = old.symbols
    fast.weights = torch.nn.ParameterList(
        [torch.nn.Parameter(w.detach().clone()) for w in old.weights])

    old_losses = _train(old, circuits, labels, epochs=3, lr=0.05)
    fast_losses = _train(fast, circuits, labels, epochs=3, lr=0.05)

    assert torch.allclose(torch.tensor(old_losses),
                          torch.tensor(fast_losses), atol=1e-4), \
        f'old={old_losses} fast={fast_losses}'
```

- [ ] **Step 2: Run the test**

Run: `$PY -m pytest tests/backend/test_fast_training.py -q` (600000ms;
loads the Bobcat model).
Expected: `1 passed`. (The autouse fixture in
`tests/backend/conftest.py` runs it with validation ON.)

If it FAILS on `allclose`: that is a real correctness finding — capture
both loss lists and report DONE_WITH_CONCERNS; do NOT loosen `atol`.
If it FAILS on `len(circuits) >= 4` or `shape == (2,)`: the first 30
captions did not yield enough uniform-shape circuits — raise the slice
from `[:30]` to `[:60]` (more captions), NOT lower the `>= 4` floor.

- [ ] **Step 3: Commit**

```bash
git add tests/backend/test_fast_training.py
git commit -m "Add training trajectory-match gate for FastPytorchModel

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Benchmark script (`benchmarks/fasttrain_bench.py`)

**Files:**
- Create: `benchmarks/fasttrain_bench.py`

Standalone script (mirrors `benchmarks/fastdiag_bench.py` /
`benchmarks/bobcat_throughput.py` conventions: positional corpus path +
flags). It trains both models and self-checks the trajectory match,
exiting nonzero on divergence so recorded numbers are never from a broken
run. It does NOT import the Task 1 test (each is self-contained).

- [ ] **Step 1: Write the script**

```python
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
    fast.symbols = old.symbols
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
```

- [ ] **Step 2: Smoke-run the script (small)**

Run: `$PY benchmarks/fasttrain_bench.py /tmp/coco_bench.txt --num 40
--epochs 3` (600000ms).
Expected: prints the kept-circuit line, a 3-row loss table with
`max abs trajectory diff` <= ~1e-4, a wall-clock table with a speedup,
and exits 0 (no divergence message). Confirm exit code with
`echo "exit=$?"`.

- [ ] **Step 3: flake8 + commit**

```bash
$PY -m flake8 benchmarks/fasttrain_bench.py
git add benchmarks/fasttrain_bench.py
git commit -m "Add train-both benchmark for the fast diagram core

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected flake8: clean (no output).

---

### Task 3: Full run + RESULTS + push

**Files:**
- Modify: `benchmarks/RESULTS.md` (append a section)

- [ ] **Step 1: Run the full benchmark**

Run: `$PY benchmarks/fasttrain_bench.py /tmp/coco_bench.txt --num 200
--epochs 10` (600000ms). Capture the full printed output (kept-circuit
line, loss table, trajectory maxdiff, wall-clock table + speedup) and
confirm exit 0. If it exits 1 (divergence), STOP and report
DONE_WITH_CONCERNS with the loss tables — do not write RESULTS from a
diverged run.

(If 200 captions is too slow to finish in one session — the stock
`PytorchModel` per-epoch contraction over ~150+ circuits is the slow
part — rerun with `--num 120` and note the actual count in RESULTS.)

- [ ] **Step 2: Append the RESULTS section**

Read `benchmarks/RESULTS.md` first to match its heading style, then
append a new section. Fill the bracketed values from the Step-1 output —
do NOT invent numbers; paste what the run printed.

```markdown
## Fast diagram core — training validation

Self-contained validation that `FastPytorchModel` trains identically to
the stock `PytorchModel` and how much faster a training epoch runs.
Script: `benchmarks/fasttrain_bench.py`. CI gate:
`tests/backend/test_fast_training.py`.

Setup: <N> COCO captions (`/tmp/coco_bench.txt`), deterministic keyword
labels, `RemoveCupsRewriter` + `SpiderAnsatz` (all atomic types ->
Dim(2)), filtered to <K> circuits with uniform output shape (2,)
(<pos> positive / <neg> negative). Both models built from the same
circuits, started from identical cloned weights, trained full-batch with
Adam (lr 0.05) for <E> epochs. i7-11800H CPU.

**Correctness:** loss trajectories matched across all <E> epochs, max abs
difference <maxdiff> (consistent with the 1.7e-6 single-backward gradient
match; the trajectory-match gate asserts allclose at atol 1e-4).

**Speed (per-epoch forward+backward+step):**

| model | ms/epoch | total |
|---|---:|---:|
| PytorchModel | <old_ms> | <old_total> s |
| FastPytorchModel | <fast_ms> | <fast_total> s |
| speedup | **<ratio>x** | |

Verdict: <one or two honest sentences — e.g. the epoch speedup tracks the
contraction share of each step; optimizer/loss overhead is common to both
and dilutes the raw contraction speedup; note if the greedy pairwise
einsum is the limiter on the larger diagrams, per the fastdiag_bench
finding.>
```

- [ ] **Step 3: Full sweep + commit + push**

```bash
$PY -m pytest tests/backend/test_fast_training.py tests/backend/test_fast_contraction.py -q
git add benchmarks/RESULTS.md
git commit -m "Record fast-core training validation results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin fast-training-validation
```

Expected: tests pass; commit made; push succeeds (the branch already
exists on origin only if pushed before — `-u` sets upstream). If push
fails on auth/remote, report it — the commits are what matter.

- [ ] **Step 4: Report**

Report the kept-circuit count, the trajectory maxdiff, the per-epoch
speedup, and whether the push succeeded.
