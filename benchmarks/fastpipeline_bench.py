#!/usr/bin/env python3
"""Benchmark the unified fast classical pipeline end-to-end.

Parses a corpus ONCE with the rust Bobcat backend, filters to a
uniform output shape (so PytorchModel can stack), then times the FULL
pipeline in two variants over all kept diagrams.

Legacy pipeline (grammar backend throughout)::

    to_diagram() -> RemoveCupsRewriter -> SpiderAnsatz
    -> PytorchModel.get_diagram_output

Fast pipeline (fast core throughout)::

    compile_fast_circuits(trees, ob_map)
        # to_fast_diagram + fast.remove_cups + FSpiderAnsatz
    -> FastPytorchModel.get_diagram_output

Both ``ob`` maps use ``Dim(2)`` for every ``AtomicType``.

Before timing, the script asserts end-to-end numeric equivalence on a
small slice (fast ``get_diagram_output`` allclose legacy at atol=1e-5
with shared weights).  Exits nonzero if not, so recorded numbers only
come from a correct build.

The timings are split into two stages:

    preprocess   everything up to (but not including) model forward
    per-step     ``get_diagram_output`` at steady state
                 (spec cache warm)

A combined line shows ``preprocess + E epochs * per-step`` total cost
for a configurable epoch count (default 50), which is the figure that
matters for real training.

Example::

    python benchmarks/fastpipeline_bench.py /tmp/coco_bench.txt
    python benchmarks/fastpipeline_bench.py /tmp/coco_bench.txt \\
        --num 400
    python benchmarks/fastpipeline_bench.py /tmp/coco_bench.txt \\
        --num 400 --epochs 100
"""
import argparse
import sys
import time
from collections import Counter

import torch

from lambeq import (AtomicType, BobcatParser, PytorchModel,
                    RemoveCupsRewriter, SpiderAnsatz, VerbosityLevel)
from lambeq.backend.fast import compile_fast_circuits
from lambeq.backend.fast.model import FastPytorchModel
from lambeq.backend.tensor import Dim


def best_of(fn, repeats):
    """Best (min) wall time of ``fn`` over ``repeats``, warm once."""
    fn()                                       # warm up (JIT/cache/alloc)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def per_diagram_ms(seconds, n):
    return seconds / n * 1000 if n else float('nan')


def main():
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument('sentence_file',
                      help='text file with one sentence per line')
    argp.add_argument('--num', type=int, default=400,
                      help='use only the first NUM sentences (default 400)')
    argp.add_argument('--repeats', type=int, default=3,
                      help='timed repeats per op (best is reported)')
    argp.add_argument('--epochs', type=int, default=50,
                      help='epoch count for the combined line (default 50)')
    args = argp.parse_args()

    with open(args.sentence_file) as f:
        sentences = [line.split() for line in f if line.strip()]
    sentences = sentences[:args.num]

    # ---- parse (rust backend; not timed) ----------------------------
    parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                          parser_backend='rust')
    trees = parser.sentences2trees(sentences, tokenised=True,
                                   suppress_exceptions=True)
    trees = [t for t in trees if t is not None]
    n_parsed = len(trees)

    print(f'command:          {" ".join(sys.argv[1:])}')
    print(f'parsed:           {n_parsed}/{len(sentences)} sentences '
          f'(rust backend)')

    # ---- modal-shape filter (identical to test_fast_pipeline.py) ----
    # SpiderAnsatz produces circuits with varying output shapes; keep
    # only the most-common shape so PytorchModel.get_diagram_output can
    # stack them into a single tensor (required for both models).
    ob = {t: Dim(2) for t in AtomicType}
    rc = RemoveCupsRewriter()
    legacy_ansatz = SpiderAnsatz(ob)

    all_g_circuits = []
    all_tree_indices = []
    for i, t in enumerate(trees):
        try:
            g = legacy_ansatz(rc(t.to_diagram()))
            all_g_circuits.append(g)
            all_tree_indices.append(i)
        except Exception:
            pass

    shapes = [tuple(c.cod.dim) for c in all_g_circuits]
    modal = Counter(shapes).most_common(1)[0][0]
    keep = [j for j, s in enumerate(shapes) if s == modal]

    g_circuits = [all_g_circuits[j] for j in keep]
    kept_trees = [trees[all_tree_indices[j]] for j in keep]
    n = len(kept_trees)

    print(f'after modal-shape filter: {n} diagrams '
          f'(shape {modal}, from {len(all_g_circuits)} ansatz-ok circuits)')

    # ---- build models for equivalence gate + timing -----------------
    legacy_model = PytorchModel.from_diagrams(g_circuits)
    torch.manual_seed(0)
    legacy_model.initialise_weights()

    # build fast circuits and model
    f_circuits = compile_fast_circuits(kept_trees, ob)
    fast_model = FastPytorchModel.from_fast_diagrams(f_circuits)
    # share exactly the same symbols / weights as the legacy model
    fast_model.symbols = legacy_model.symbols
    fast_model.weights = legacy_model.weights

    # ---- numeric equivalence gate (small slice) ----------------------
    gate_n = min(10, n)
    expected = legacy_model.get_diagram_output(g_circuits[:gate_n])
    # warm the fast spec cache for the gate slice
    _ = fast_model.get_diagram_output(f_circuits[:gate_n])
    got = fast_model.get_diagram_output(f_circuits[:gate_n])
    mismatches = 0
    for i in range(gate_n):
        if not torch.allclose(got[i], expected[i], atol=1e-5):
            print(f'  NUMERIC MISMATCH at diagram {i}', file=sys.stderr)
            mismatches += 1
    if mismatches:
        print(f'numeric gate FAILED: {mismatches}/{gate_n} mismatches',
              file=sys.stderr)
        sys.exit(1)
    print(f'numeric gate:     PASS ({gate_n}/{gate_n} allclose at atol=1e-5)')

    # ---- timed closures ---------------------------------------------
    r = args.repeats

    # Legacy preprocess: to_diagram + RemoveCupsRewriter + SpiderAnsatz
    # (re-runs the full preprocess to measure cost, not the warm copies)
    def legacy_preprocess():
        ans = SpiderAnsatz(ob)
        rw = RemoveCupsRewriter()
        return [ans(rw(t.to_diagram())) for t in kept_trees]

    # Fast preprocess: compile_fast_circuits (to_fast_diagram +
    # fast.remove_cups + FSpiderAnsatz — all on the fast core)
    def fast_preprocess():
        return compile_fast_circuits(kept_trees, ob)

    # Legacy per-step: get_diagram_output on the grammar circuits
    def legacy_step():
        return legacy_model.get_diagram_output(g_circuits)

    # Fast per-step: get_diagram_output at steady state (cache warm).
    # The cache is populated during the warm-up call inside best_of().
    def fast_step():
        return fast_model.get_diagram_output(f_circuits)

    # ---- time -------------------------------------------------------
    lp_s = best_of(legacy_preprocess, r)
    fp_s = best_of(fast_preprocess, r)
    ls_s = best_of(legacy_step, r)
    fs_s = best_of(fast_step, r)

    lp_ms = per_diagram_ms(lp_s, n)
    fp_ms = per_diagram_ms(fp_s, n)
    ls_ms = per_diagram_ms(ls_s, n)
    fs_ms = per_diagram_ms(fs_s, n)

    preprocess_speedup = lp_ms / fp_ms if fp_ms else float('inf')
    step_speedup = ls_ms / fs_ms if fs_ms else float('inf')

    # Combined: preprocess (once) + E epochs × per-step (per diagram).
    E = args.epochs
    legacy_combined = lp_ms + E * ls_ms
    fast_combined = fp_ms + E * fs_ms
    combined_speedup = (legacy_combined / fast_combined
                        if fast_combined else float('inf'))

    # ---- results table ----------------------------------------------
    print()
    print(f'{"stage":<26}{"legacy ms":>12}{"fast ms":>10}{"speedup":>10}'
          f'{"n":>7}')
    print('-' * 65)
    print(f'{"preprocess":<26}{lp_ms:>12.4f}{fp_ms:>10.4f}'
          f'{preprocess_speedup:>9.1f}x{n:>7}')
    print(f'{"per-step (get_diag_out)":<26}{ls_ms:>12.4f}{fs_ms:>10.4f}'
          f'{step_speedup:>9.1f}x{n:>7}')
    print('-' * 65)
    print(f'combined preprocess + {E} epochs  --  '
          f'legacy {legacy_combined:.2f} ms  vs  '
          f'fast {fast_combined:.2f} ms  =  {combined_speedup:.1f}x')
    print()
    print(f'repeats: {r} (best of {r}, 1 warm-up); '
          f'ob = Dim(2) for every AtomicType')
    print()
    print('Notes:')
    print('  preprocess legacy = to_diagram + RemoveCupsRewriter'
          ' + SpiderAnsatz')
    print('  preprocess fast   = compile_fast_circuits (to_fast_diagram +')
    print('                      fast.remove_cups + FSpiderAnsatz)')
    print('  per-step fast     = FastPytorchModel at steady state '
          '(spec cache warm;')
    print('                      spec extracted once per distinct diagram)')
    print('  Timing is post-parser only.  On CPU the BobcatParser tagger')
    print('  dominates end-to-end wall time, so these numbers are most')
    print('  relevant for the GPU / large-batch regime where parsing is')
    print('  amortised or parallelised.')


if __name__ == '__main__':
    main()
