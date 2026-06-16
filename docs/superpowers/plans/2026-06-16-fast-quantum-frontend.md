# Fast Quantum Front-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `compile_quantum_input(trees)` helper producing the cup-reduced `grammar.Diagram` any `CircuitAnsatz` consumes, built via the fast front-end (fast construction + fast `remove_cups`), gated byte-identical to the legacy front-end and to the resulting quantum circuit.

**Architecture:** `to_grammar(remove_cups(t.to_fast_diagram()))` == `RemoveCupsRewriter()(t.to_diagram())` (composition of two existing byte-identical gates on `fast-pipeline`), so the quantum ansatz/backends run unchanged on the fast-built input. Front-end only.

**Tech Stack:** Pure Python. Fast core on `fast-pipeline` (`to_fast_diagram`, `remove_cups`, `convert.to_grammar`). Oracles: `RemoveCupsRewriter`, `IQPAnsatz`.

**Spec:** `docs/superpowers/specs/2026-06-16-fast-quantum-frontend-design.md`

---

## Conventions

- Repo `/home/kinianlo/projects/lambeq`, branch `fast-quantum-frontend`
  (off `fast-pipeline`). `PY=~/.pyenv/versions/qnlp/bin/python`. 600000ms
  timeouts on Bobcat-loading commands. Corpus `/tmp/coco_bench.txt`.
- Commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Additive: nothing in `CircuitAnsatz`/quantum models/`RemoveCupsRewriter`
  changes.

## Verified facts

- `CircuitAnsatz.__call__(diagram: grammar.Diagram) -> Circuit` (a quantum
  `Diagram`); `IQPAnsatz(ob_map: Mapping[Ty, int], n_layers: int, ...)` —
  ob_map maps `Ty` → qubit count (int); `n_layers` required.
- Quantum circuits support structural `==` (`IQPAnsatz(...)(d) ==
  IQPAnsatz(...)(d)` is True for identical input).
- `lambeq.backend.fast.compile_fast_circuits` lives in `pipeline.py`;
  `convert.to_grammar`, `remove_cups`, `CCGTree.to_fast_diagram` all on
  this branch.

---

### Task 1: `compile_quantum_input` helper + gates

**Files:** modify `lambeq/backend/fast/pipeline.py`,
`lambeq/backend/fast/__init__.py`; Test
`tests/backend/test_fast_quantum_frontend.py`.

- [ ] **Step 1: implement** — add to `lambeq/backend/fast/pipeline.py`:

```python
def compile_quantum_input(trees):
    """Cup-reduced grammar.Diagrams for any CircuitAnsatz, built via the
    fast front-end (fast construction + fast remove_cups).

    Equivalent to ``[RemoveCupsRewriter()(t.to_diagram()) for t in
    trees]`` but built on the fast core, so quantum preprocessing gets
    the construction + cup-removal speedups. The quantum ansatz and
    backends consume the result unchanged.
    """
    from lambeq.backend.fast import convert
    from lambeq.backend.fast.normal import remove_cups
    return [convert.to_grammar(remove_cups(t.to_fast_diagram()))
            for t in trees]
```

(Match the existing import style in pipeline.py — if `convert`/
`remove_cups` are already imported at module top there, use those.)
Re-export `compile_quantum_input` from `lambeq/backend/fast/__init__.py`
(add to the `from lambeq.backend.fast.pipeline import ...` line and
`__all__`).

- [ ] **Step 2: gates** (`tests/backend/test_fast_quantum_frontend.py`):

```python
import pytest

from lambeq import AtomicType, IQPAnsatz, RemoveCupsRewriter
from lambeq.backend.fast import compile_quantum_input

N, S = AtomicType.NOUN, AtomicType.SENTENCE
QOB = {N: 1, S: 1}


def test_quantum_frontend_matches_legacy(bobcat_trees):
    assert bobcat_trees
    rc = RemoveCupsRewriter()
    fast = compile_quantum_input(bobcat_trees)
    for t, fd in zip(bobcat_trees, fast):
        assert fd == rc(t.to_diagram()), t


def test_iqp_circuit_equivalent(bobcat_trees):
    rc = RemoveCupsRewriter()
    ansatz = IQPAnsatz(QOB, n_layers=1)
    fast = compile_quantum_input(bobcat_trees)
    for t, fd in zip(bobcat_trees, fast):
        legacy_circ = ansatz(rc(t.to_diagram()))
        fast_circ = ansatz(fd)
        assert fast_circ == legacy_circ, t
```

(Gate 1 — front-end identity — already follows from the composition, but
assert it. Gate 2 — the actual quantum circuit is byte-identical. If
`==` on circuits proves unreliable, compare `repr(fast_circ) ==
repr(legacy_circ)`; the substance is circuit identity, do not weaken.)

- [ ] **Step 3: run + flake8 + commit** `$PY -m pytest
tests/backend/test_fast_quantum_frontend.py -q` (600000ms; loads Bobcat)
→ both gates pass over the corpus. `$PY -m flake8
lambeq/backend/fast/pipeline.py tests/backend/test_fast_quantum_frontend.py`.

```bash
git add lambeq/backend/fast/pipeline.py lambeq/backend/fast/__init__.py tests/backend/test_fast_quantum_frontend.py
git commit -m "Add compile_quantum_input: fast front-end for the quantum pipeline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

If gate 2 FAILS (circuits differ): the fast front-end diverges from the
legacy front-end for some diagram — but gate 1 should catch that first
(if gate 1 passes and gate 2 fails, it's an ansatz-determinism issue,
investigate; do NOT weaken). The per-component construction + remove_cups
gates already prove the front-end identity, so a failure here is
surprising — report DONE_WITH_CONCERNS with the diagram.

---

### Task 2: Front-end benchmark + RESULTS + push

**Files:** Create `benchmarks/fastquantum_bench.py`; modify
`benchmarks/RESULTS.md`.

- [ ] **Step 1: write `benchmarks/fastquantum_bench.py`** — parse the
corpus (rust backend), then time the QUANTUM FRONT-END only, over all
trees (best-of-3, warm-up, argparse positional corpus + `--num`):
- legacy: `[RemoveCupsRewriter()(t.to_diagram()) for t in trees]`
- fast: `compile_quantum_input(trees)`
Report ms/diagram legacy vs fast + speedup. BEFORE timing, assert
`compile_quantum_input(trees[:n])[i] == RemoveCupsRewriter()(trees[i].to_diagram())`
for a few trees (so numbers come from a correct build). Structure like
`benchmarks/fastpipeline_bench.py`.

- [ ] **Step 2: run** `$PY benchmarks/fastquantum_bench.py
/tmp/coco_bench.txt --num 400` (note count). Capture the table.

- [ ] **Step 3: append `## Fast quantum front-end` to
`benchmarks/RESULTS.md`** — legacy vs fast front-end ms/diagram +
speedup; note this is the SAME construction + remove_cups win carried to
quantum preprocessing; the ansatz + circuit execution (pennylane/tket)
are unchanged and NOT measured. Honest caveat: this accelerates quantum
diagram-building, not circuit execution (which dominates QML wall-clock).
PASTE REAL NUMBERS.

- [ ] **Step 4: full sweep + commit + push**

```bash
$PY -m pytest tests/backend/test_fast_quantum_frontend.py tests/backend/test_fast_remove_cups.py tests/backend/test_fast_construct.py -q
$PY -m flake8 benchmarks/fastquantum_bench.py
git add benchmarks/fastquantum_bench.py benchmarks/RESULTS.md
git commit -m "Benchmark the fast quantum front-end

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin fast-quantum-frontend
```

- [ ] **Step 5: report** — the two gate results (front-end identity +
IQP circuit equivalence, N/N), the front-end speedup, whether push
succeeded.
