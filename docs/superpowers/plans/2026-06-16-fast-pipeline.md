# Unified Fast Pipeline (Tier 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One branch running the full fast classical pipeline (parse → `to_fast_diagram` → fast `remove_cups` → `FSpiderAnsatz` → `FastPytorchModel`), numerically identical to the legacy pipeline, with the compounded throughput measured.

**Architecture:** `fast-pipeline` already merges direct construction + `FSpiderAnsatz`. This plan adds the last fast stage — a fast-core `remove_cups` (`FDiagram` port of `RemoveCupsRewriter`, keeps cup-removal for quantum *and* makes it fast) — makes `FastPytorchModel` consume fast `FDiagram` circuits natively, adds a thin compile helper, and gates the whole pipeline end-to-end against the legacy one.

**Tech Stack:** Pure Python. Oracles: `RemoveCupsRewriter` (rewrite/rewrite_diagram.py:108-192), the full legacy pipeline (`to_diagram → RemoveCups → SpiderAnsatz → PytorchModel`). Fast core: `lambeq/backend/fast/` (FDiagram/FBox/FTy ops, `FSpiderAnsatz`, `contraction`, `FastPytorchModel`).

**Spec:** `docs/superpowers/specs/2026-06-16-fast-pipeline-design.md`

---

## Conventions (every task)

- Repo `/home/kinianlo/projects/lambeq`, branch `fast-pipeline`.
  `PY=~/.pyenv/versions/qnlp/bin/python`. 600000ms timeouts on Bobcat-
  loading commands. Corpus `/tmp/coco_bench.txt`.
- Commit style: plain sentence + trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Additive: the legacy `RemoveCupsRewriter`, `SpiderAnsatz`,
  `PytorchModel` are NOT modified (they are oracles). Files touched:
  `lambeq/backend/fast/normal.py` (+`remove_cups`), `model.py` (native
  path), new `pipeline.py`, `__init__.py` re-exports, tests, benchmark.
- Tests run with validation ON (conftest).

## Fast-core API (verified)

- `FDiagram`: `id(ty)`, `then`/`>>`, `tensor`/`@`, `dagger()`,
  `.boxes`, `.offsets`, `.dom`, `.cod`, `.terms` (tuple of `(FBox, off)`).
- `FBox`: `name, dom, cod, kind, z, is_dagger, payload`; `dagger()`.
- `FTy`: `@`, `.l`, `.r`, `len`, slicing `[i:j]`.
- `fast.diagram`: `cup, cap, swap, spider, word, cups, caps`, kinds
  `CUP/CAP/SWAP/SPIDER/WORD/PLAIN`.
- `convert.to_fast(grammar.Diagram)`, `convert.to_grammar(FDiagram)`.
- `normal.py` working-list pattern: `Term = tuple[FBox, int]`, helpers
  operate on `list[Term]`; one `FDiagram` built at the end.
- `FastPytorchModel` (model.py): `_spec_for(diagram)` does
  `to_contraction(convert.to_fast(diagram))`; `get_diagram_output` stacks
  `evaluate(self._spec_for(d), params)`; `from_diagrams`/
  `initialise_weights` inherited from `PytorchModel`.
- `PytorchModel.from_diagrams` collects symbols as
  `sorted({s for c in circuits for s in c.free_symbols})`.

---

### Task 1: Fast `remove_cups` (`lambeq/backend/fast/normal.py`)

Port `RemoveCupsRewriter` onto `FDiagram`. **This is the hardest task**
(comparable to the snake-removal port already in this file). The legacy
(rewrite/rewrite_diagram.py:108-192) is
`rewrite = _remove_cups(_compress_cups(_remove_cups(d)))`:

- `_compress_cups(d)`: single pass over `(box, offset)`; merge an
  adjacent nested cup (`isinstance(box, Cup)` and prev layer is a Cup at
  `offset == prev_offset - 1`) into one wide `Box(CUP_TOKEN, dom, Ty())`
  where `dom = box.dom[:1] @ prev.dom @ box.dom[1:]`; else append. Rebuild
  via `then_at`.
- `_remove_cups(d)`: greedy contraction. Maintain `diags: list[Diagram|
  Box]` starting `[Id(dom)]`. For each `(box, offset)`: find the partial
  `diags[i]` the offset lands in; if `off==0 and not box.dom` insert the
  box; else gather `left=diags[i]`, extend `right` with following diags
  until `len(left.cod @ right.cod) >= off + len(box.dom)`; then for a Cup
  contract by daggering+bending (`left.dagger().r @ wires_r` etc., with
  the `pg_type1.r == pg_type2` reversed/illegal branches) or combine via
  `Diagram.cups`; non-cup boxes compose `left @ right >> wires_l @ box @
  wires_r`. Return `Id().tensor(*diags)`.

**Files:** modify `lambeq/backend/fast/normal.py` (add `remove_cups`),
`lambeq/backend/fast/__init__.py` (re-export); Test
`tests/backend/test_fast_remove_cups.py`.

**Contract:** `remove_cups(d: FDiagram) -> FDiagram` porting the three-pass
structure onto FDiagram fragments. The fast equivalents exist: `CUP_TOKEN`
→ an `FBox` of kind PLAIN named with the same token string (so
`to_grammar` round-trips it; verify the legacy `CUP_TOKEN` constant and
match it), `Id(ty)`→`FDiagram.id`, `@`→`tensor`, `>>`→`then`,
`box.dagger()`→`FBox.dagger`/`FDiagram.dagger`, `ty.r/.l`→`FTy.r/.l`,
`Diagram.cups(a,b,is_reversed=...)`→ build via `fast.diagram` cup boxes
(mirror `cups`/the reversed orientation). `then_at(box, off)` →
`FDiagram.id(frontier[:off]) @ box.to_diagram() @ FDiagram.id(
frontier[off+len(box.dom):])` composed with `>>` (express it; there is no
`then_at` on FDiagram — add a small local helper).

- [ ] **Step 1: failing tests** (`tests/backend/test_fast_remove_cups.py`):

```python
from lambeq import RemoveCupsRewriter
from lambeq.backend.fast import convert
from lambeq.backend.fast.normal import remove_cups


def test_remove_cups_matches_oracle(bobcat_diagrams):
    assert bobcat_diagrams
    rc = RemoveCupsRewriter()
    for d in bobcat_diagrams:
        oracle = rc(d)
        fast = convert.to_grammar(remove_cups(convert.to_fast(d)))
        assert fast == oracle, d


def test_remove_cups_reduces_cups(bobcat_diagrams):
    from lambeq.backend.grammar import Cup
    rc = RemoveCupsRewriter()
    for d in bobcat_diagrams[:15]:
        before = sum(isinstance(b, Cup) for b in d.boxes)
        after_fast = convert.to_grammar(remove_cups(convert.to_fast(d)))
        after = sum(isinstance(b, Cup) for b in after_fast.boxes)
        assert after <= before
```

Also add a couple of hand-built small cases (a single cup state; nested
cups) asserting `to_grammar(remove_cups(to_fast(d))) == rc(d)`.

- [ ] **Step 2: fail → implement → green.** `$PY -m pytest
tests/backend/test_fast_remove_cups.py -q` (600000ms). The differential
`== RemoveCupsRewriter()(d)` over the corpus is THE gate. Debug a mismatch
by diffing layer reprs of the first failing diagram and tracing the legacy
vs fast contraction order.

**FALLBACK (if byte-identical proves intractable):** the PURPOSE is cup
removal (fewer cups, same meaning), not byte-identity to the specific
legacy algorithm. If the exact bend ordering diverges in ways that are
still valid cup-removals, switch `test_remove_cups_matches_oracle` to a
NUMERIC + structural gate: (a) `to_grammar(remove_cups(to_fast(d)))`
contracts to the same tensor as `rc(d)` (build both into PytorchModel-
style circuits or use `tensornetwork` eval with fixed random box arrays,
allclose 1e-5), AND (b) cup count not increased. Document which gate
shipped in the test docstring and report it. Do NOT ship without one of
the two gates passing over the corpus.

- [ ] **Step 3: commit**

```bash
git add lambeq/backend/fast/normal.py lambeq/backend/fast/__init__.py tests/backend/test_fast_remove_cups.py
git commit -m "Add fast-core remove_cups (FDiagram port of RemoveCupsRewriter)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Native fast model path (`lambeq/backend/fast/model.py`)

**Files:** modify `lambeq/backend/fast/model.py`; Test add to
`tests/backend/test_fast_pipeline.py`.

- [ ] **Step 1: implement.** Make `FastPytorchModel` accept fast
`FDiagram` circuits directly:

```python
    def _spec_for(self, diagram):
        from lambeq.backend.fast.diagram import FDiagram
        key = id(diagram)
        hit = self._spec_cache.get(key)
        if hit is None or hit[0] is not diagram:
            fd = diagram if isinstance(diagram, FDiagram) \
                else convert.to_fast(diagram)
            spec = contraction.to_contraction(fd)
            self._spec_cache[key] = (diagram, spec)
            return spec
        return hit[1]

    @classmethod
    def from_fast_diagrams(cls, circuits, **kwargs):
        """Build from fast FDiagram circuits (FSpiderAnsatz output).
        Collects Symbol payloads as the model's symbols."""
        from lambeq.backend.symbol import Symbol
        model = cls(**kwargs)
        syms = set()
        for c in circuits:
            for b, _ in c.terms:
                if isinstance(b.payload, Symbol):
                    syms.add(b.payload)
        model.symbols = sorted(syms, key=lambda s: s.name)
        return model
```

(Match `convert`/`contraction` imports already at the top of model.py.
Verify the legacy `from_diagrams` sort key — `PytorchModel.from_diagrams`
sorts symbols; mirror its ordering so weight indexing matches. If it
sorts by the Symbol's natural order, use that instead of `name`.)

- [ ] **Step 2: unit tests** (in `tests/backend/test_fast_pipeline.py`):

```python
import pytest

from lambeq import AtomicType, RemoveCupsRewriter, SpiderAnsatz
from lambeq.backend.fast import FSpiderAnsatz, convert
from lambeq.backend.fast.model import FastPytorchModel
from lambeq.backend.tensor import Dim

OB = {t: Dim(2) for t in AtomicType}


def test_from_fast_diagrams_symbols_match(bobcat_diagrams):
    from lambeq import PytorchModel
    rc = RemoveCupsRewriter()
    ds = bobcat_diagrams[:10]
    legacy = PytorchModel.from_diagrams([SpiderAnsatz(OB)(rc(d)) for d in ds])
    fans = FSpiderAnsatz(OB)
    fast = FastPytorchModel.from_fast_diagrams(
        [fans(convert.to_fast(rc(d))) for d in ds])
    assert {s.name for s in fast.symbols} == {s.name for s in legacy.symbols}


def test_spec_for_accepts_fdiagram(bobcat_diagrams):
    fans = FSpiderAnsatz(OB)
    fd = fans(convert.to_fast(bobcat_diagrams[0]))
    m = FastPytorchModel.from_fast_diagrams([fd])
    # _spec_for must NOT call convert.to_fast on an FDiagram (no raise)
    spec = m._spec_for(fd)
    assert spec is not None
```

- [ ] **Step 3: run + commit** `$PY -m pytest tests/backend/test_fast_pipeline.py -q` (600000ms) → green.

```bash
git add lambeq/backend/fast/model.py tests/backend/test_fast_pipeline.py
git commit -m "Let FastPytorchModel consume fast FDiagram circuits natively

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Pipeline helper + end-to-end numeric gate

**Files:** Create `lambeq/backend/fast/pipeline.py`; modify
`lambeq/backend/fast/__init__.py`; add the gate to
`tests/backend/test_fast_pipeline.py`.

- [ ] **Step 1: implement `lambeq/backend/fast/pipeline.py`:**

```python
# (Apache header)
"""One-call fast classical pipeline: trees -> tensorised FDiagrams."""

from __future__ import annotations

from collections.abc import Mapping

from lambeq.backend import grammar
from lambeq.backend.fast.ansatz import FSpiderAnsatz
from lambeq.backend.fast.normal import remove_cups
from lambeq.backend.tensor import Dim


def compile_fast_circuits(trees, ob_map: Mapping[grammar.Ty, Dim],
                          max_order: int = 2):
    """trees -> [FSpiderAnsatz(remove_cups(t.to_fast_diagram()))]."""
    ansatz = FSpiderAnsatz(ob_map, max_order)
    return [ansatz(remove_cups(t.to_fast_diagram())) for t in trees]
```

Re-export `compile_fast_circuits` from `__init__.py`.

- [ ] **Step 2: end-to-end numeric gate** (the headline; add to
`tests/backend/test_fast_pipeline.py`):

```python
def test_pipeline_matches_legacy_end_to_end(bobcat_trees):
    torch = pytest.importorskip('torch')
    from collections import Counter
    from lambeq import PytorchModel
    from lambeq.backend.fast import compile_fast_circuits

    rc = RemoveCupsRewriter()
    diagrams = [t.to_diagram() for t in bobcat_trees]
    g_circuits = [SpiderAnsatz(OB)(rc(d)) for d in diagrams]
    # filter to uniform output shape so PytorchModel can stack
    shapes = [tuple(c.cod.dim) for c in g_circuits]
    modal = Counter(shapes).most_common(1)[0][0]
    keep = [i for i, s in enumerate(shapes) if s == modal]
    trees = [bobcat_trees[i] for i in keep]
    g_circuits = [g_circuits[i] for i in keep]
    assert len(keep) >= 4

    legacy = PytorchModel.from_diagrams(g_circuits)
    torch.manual_seed(0)
    legacy.initialise_weights()
    expected = legacy.get_diagram_output(g_circuits)
    weights = dict(zip(legacy.symbols, legacy.weights))

    f_circuits = compile_fast_circuits(trees, OB)
    fast = FastPytorchModel.from_fast_diagrams(f_circuits)
    fast.symbols = legacy.symbols
    fast.weights = legacy.weights          # share exact weights
    got = fast.get_diagram_output(f_circuits)
    assert torch.allclose(got, expected, atol=1e-5)
```

This proves the FULL fast pipeline (construct + remove_cups + ansatz +
contraction) equals the FULL legacy pipeline. (`bobcat_trees` fixture
already exists in conftest.)

- [ ] **Step 3: run + commit** `$PY -m pytest tests/backend/test_fast_pipeline.py -q` (600000ms) → all green.

```bash
git add lambeq/backend/fast/pipeline.py lambeq/backend/fast/__init__.py tests/backend/test_fast_pipeline.py
git commit -m "Add compile_fast_circuits helper and end-to-end pipeline gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Compounded benchmark + RESULTS + push

**Files:** Create `benchmarks/fastpipeline_bench.py`; modify
`benchmarks/RESULTS.md`.

- [ ] **Step 1: write `benchmarks/fastpipeline_bench.py`** — parse the
corpus (rust backend), filter to a uniform output shape, then time the
FULL pipelines:
- legacy preprocess: `[SpiderAnsatz(OB)(rc(t.to_diagram())) for t in trees]`
- fast preprocess: `compile_fast_circuits(trees, OB)`
- legacy per-step: `PytorchModel.get_diagram_output(g_circuits)`
- fast per-step: `FastPytorchModel.from_fast_diagrams(f).get_diagram_output(f)`
  (warm once to build the spec cache, then time the steady state)
Report preprocess ms/diagram + per-step ms/diagram, legacy vs fast +
speedup, and a combined "preprocess + E epochs" line. Best-of-3, warm-up,
argparse + `--num`, `--epochs`. BEFORE timing, assert the end-to-end
numeric equivalence (a few diagrams) so numbers come from a correct build.
Structure like `benchmarks/fastdiag_bench.py`.

- [ ] **Step 2: run** `$PY benchmarks/fastpipeline_bench.py
/tmp/coco_bench.txt --num 400` (note count). Capture the table.

- [ ] **Step 3: append `## Unified fast pipeline (Tier 1)` to
`benchmarks/RESULTS.md`** — the measured preprocess + per-step + combined
speedups (legacy vs fast), the fact that the fast pipeline now includes
fast `remove_cups` (no skipped stage), and the honest caveats: post-parser
only; laptop-CPU parser dominates off-GPU so this win lands in the GPU/
large-scale regime. PASTE REAL NUMBERS.

- [ ] **Step 4: full sweep + commit + push**

```bash
$PY -m pytest tests/backend/test_fast_remove_cups.py tests/backend/test_fast_pipeline.py tests/backend/test_fast_ansatz.py tests/backend/test_fast_construct.py tests/backend/test_fast_contraction.py -q
$PY -m flake8 lambeq/backend/fast/normal.py lambeq/backend/fast/model.py lambeq/backend/fast/pipeline.py benchmarks/fastpipeline_bench.py
git add benchmarks/fastpipeline_bench.py benchmarks/RESULTS.md
git commit -m "Benchmark the unified fast pipeline end to end

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin fast-pipeline
```

- [ ] **Step 5: report** — fast `remove_cups` gate result (byte-identical
or numeric fallback, N/N), the compounded preprocess + per-step speedups,
whether push succeeded.
