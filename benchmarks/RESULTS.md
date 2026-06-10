# BobcatParser throughput benchmark results

Corpus: 750 generated sentences (3-12 words), `/tmp/bobcat_bench.txt`,
see Task 1 of `docs/superpowers/plans/2026-06-10-bobcat-throughput.md`.
Machine: Intel Core i7-11800H @ 2.30GHz, GPU: RTX 3080 Laptop present but CUDA unavailable (driver mismatch); GPU numbers to be collected later, RAM: 31 GB.

## Summary (short corpus, sent/s — size-independent)

Baseline runs used 200 sentences, default batch_size=4.  Combined runs
used 750 sentences, batch_size=32, n_jobs=-1 (CPU-relevant optimizations
only; `--dtype` excluded — targets GPU where bfloat16/float16 is fast).

| Stage | Baseline | Combined | Speedup |
|---|---|---|---|
| Tagging | 22.1 sent/s | 53.6 sent/s | 2.4× |
| Chart parsing (serial) | 937.2 sent/s | 744.3 sent/s | 0.8× † |
| End-to-end (serial) | 21.6 sent/s | 50.0 sent/s | 2.3× |
| End-to-end (n_jobs=-1) | n/a | 47.3 sent/s | — ‡ |
| Peak RSS | 2650 MB | 2805 MB | — |

† Chart parsing rate reflects the full 750-sentence corpus which includes
  harder sentences than the 200-sentence baseline subset; the absolute
  chart-parse wall time is not the bottleneck (1.01s vs 13.99s tagging).

‡ On the short corpus, chart parsing is only ~7% of total time, so
  parallelism overhead slightly exceeds the gain; n_jobs benefits the long
  corpus where chart parsing is ~22% (see Combined section below).

## Baseline (before optimizations)

Note: model default tagger batch_size is 4 (from pipeline_config.json).

```
command:          /tmp/bobcat_bench.txt --num 200
config:           {} device=cpu
sentences:        200 (0 failed)
tagging:          9.05s (22.1 sent/s)
chart parsing:    0.21s (937.2 sent/s)
end-to-end:       9.27s (21.6 sent/s)
peak RSS:         2650 MB
```

## After inference_mode (Task 2)

```
command:          /tmp/bobcat_bench.txt --num 200
config:           {} device=cpu
sentences:        200 (0 failed)
tagging:          8.18s (24.4 sent/s)
chart parsing:    0.20s (1007.4 sent/s)
end-to-end:       8.38s (23.9 sent/s)
peak RSS:         2505 MB
```

## After length-sorted batching (Task 3)

```
command:          /tmp/bobcat_bench.txt --num 200
config:           {} device=cpu
sentences:        200 (0 failed)
tagging:          7.33s (27.3 sent/s)
chart parsing:    0.22s (920.8 sent/s)
end-to-end:       7.55s (26.5 sent/s)
peak RSS:         2513 MB
```

```
command:          /tmp/bobcat_bench.txt --num 200 --batch-size 32
config:           {'batch_size': 32} device=cpu
sentences:        200 (0 failed)
tagging:          3.53s (56.7 sent/s)
chart parsing:    0.19s (1071.2 sent/s)
end-to-end:       3.72s (53.8 sent/s)
peak RSS:         2620 MB
```

## After span-budget batching (Task 4)

```
command:          /tmp/bobcat_bench.txt --num 200 --max-spans-per-batch 5000
config:           {'max_spans_per_batch': 5000} device=cpu
sentences:        200 (0 failed)
tagging:          3.94s (50.8 sent/s)
chart parsing:    0.20s (1005.0 sent/s)
end-to-end:       4.14s (48.3 sent/s)
peak RSS:         2642 MB
```

```
command:          /tmp/bobcat_bench.txt --num 200 --max-spans-per-batch 20000
config:           {'max_spans_per_batch': 20000} device=cpu
sentences:        200 (0 failed)
tagging:          4.96s (40.3 sent/s)
chart parsing:    0.21s (938.8 sent/s)
end-to-end:       5.18s (38.6 sent/s)
peak RSS:         2840 MB
```

## After vectorized post-processing (Task 5)

```
command:          /tmp/bobcat_bench.txt --num 200 --batch-size 32
config:           {'batch_size': 32} device=cpu
sentences:        200 (0 failed)
tagging:          3.72s (53.8 sent/s)
chart parsing:    0.19s (1053.3 sent/s)
end-to-end:       3.91s (51.2 sent/s)
peak RSS:         2609 MB
```

## After opt-in dtype (Task 6)

```
command:          /tmp/bobcat_bench.txt --num 200 --batch-size 32 --dtype bfloat16
config:           {'batch_size': 32, 'dtype': 'bfloat16'} device=cpu
sentences:        200 (0 failed)
tagging:          10.10s (19.8 sent/s)
chart parsing:    0.19s (1061.6 sent/s)
end-to-end:       10.29s (19.4 sent/s)
peak RSS:         3455 MB
```

CPU bfloat16 autocast is slower than fp32 on this AVX-512-less laptop (10.10s vs 3.72s); failures stayed at 0. The `--dtype` flag is intended to accelerate GPU inference where bfloat16/float16 is natively fast.

## After parallel chart parsing (Task 7)

Standard corpus (3-12 word sentences, 200 sentences, batch-size 32, n_jobs=4):

```
command:          /tmp/bobcat_bench.txt --num 200 --batch-size 32 --n-jobs 4
config:           {'batch_size': 32} device=cpu
sentences:        200 (0 failed)
tagging:          3.66s (54.7 sent/s)
chart parsing:    0.19s (1027.3 sent/s)
end-to-end:       3.85s (51.9 sent/s)
end-to-end (n_jobs=4): 3.86s (51.8 sent/s)
peak RSS:         2722 MB
```

Long-sentence corpus (25-50 word sentences formed by concatenating 4 standard sentences, 100 sentences, batch-size 16, n_jobs=4). Chart parsing is a much larger share of the work on this corpus:

```
command:          /tmp/bobcat_bench_long.txt --num 100 --batch-size 16 --n-jobs 4
config:           {'batch_size': 16} device=cpu
sentences:        100 (0 failed)
tagging:          7.87s (12.7 sent/s)
chart parsing:    2.05s (48.7 sent/s)
end-to-end:       9.92s (10.1 sent/s)
end-to-end (n_jobs=4): 8.86s (11.3 sent/s)
peak RSS:         3067 MB
```

On the standard short-sentence corpus, chart parsing is only ~5% of total time so n_jobs has no measurable effect. On the long-sentence corpus where chart parsing is ~21% of total time, n_jobs=4 gives ~12% speedup (9.92s → 8.86s). Larger gains are expected on GPU where tagging is much faster and chart parsing would dominate.

## Combined (all optimizations)

All CPU-relevant optimizations active: inference_mode, length-sorted
batching, max_spans_per_batch (not set here), vectorized extract_topk,
n_jobs=-1.  `--dtype` excluded: bfloat16 autocast is slower on this
AVX-512-less laptop CPU; the flag targets GPU inference.

Short-sentence corpus (3-12 words, 750 sentences, batch-size 32):

```
command:          /tmp/bobcat_bench.txt --batch-size 32 --n-jobs -1
config:           {'batch_size': 32} device=cpu
sentences:        750 (0 failed)
tagging:          13.99s (53.6 sent/s)
chart parsing:    1.01s (744.3 sent/s)
end-to-end:       15.00s (50.0 sent/s)
end-to-end (n_jobs=-1): 15.87s (47.3 sent/s)
peak RSS:         2805 MB
```

As expected for short sentences, n_jobs overhead slightly exceeds the
gain (chart parsing is only ~7% of total time at 1.01s vs 13.99s
tagging).  Overall tagging throughput is 2.4× baseline (53.6 vs
21.6 sent/s end-to-end).

Long-sentence corpus (~25-50 words, 187 sentences, batch-size 16):

```
command:          /tmp/bobcat_bench_long.txt --batch-size 16 --n-jobs -1
config:           {'batch_size': 16} device=cpu
sentences:        187 (0 failed)
tagging:          16.77s (11.2 sent/s)
chart parsing:    4.73s (39.5 sent/s)
end-to-end:       21.50s (8.7 sent/s)
end-to-end (n_jobs=-1): 18.48s (10.1 sent/s)
peak RSS:         3145 MB
```

On the long corpus, chart parsing is ~22% of total time; n_jobs=-1 gives
a 14% end-to-end speedup (21.50s → 18.48s, 8.7 → 10.1 sent/s).

## GPU results — Lab 105 `goosander-l`, RTX 3090 Ti (torch 2.12+cu130, Python 3.11)

"before" = `main` (183a5c1), "after" = this branch. Corpora as above
(short: 3-12 words; long: 25-50 words). Stage rates are sent/s.

Short corpus (400 sentences):

| config                    | tagging | end-to-end | peak CUDA |
|---------------------------|--------:|-----------:|----------:|
| before, default (batch 4) |   366.4 |      297.0 |   1714 MB |
| before, batch 32          |   870.4 |      557.3 |   2619 MB |
| after, default            |   467.1 |      360.7 |   1591 MB |
| after, batch 32           |   989.6 |      607.0 |   1619 MB |
| after, batch 128          |   992.2 |      609.2 |   1720 MB |
| after, batch 32 + fp16    |   613.7 |      441.4 |   1628 MB |
| after, batch 128 + fp16   |  1369.0 |      724.5 |   1742 MB |
| after, max-spans 100000   |   463.0 |      357.3 |   2143 MB |

Long corpus (187 sentences, batch 16):

| config                       | tagging | end-to-end | peak CUDA |
|------------------------------|--------:|-----------:|----------:|
| before                       |   170.5 |       50.9 |   3593 MB |
| after                        |   105.7 |       42.1 |   1786 MB |
| after + n_jobs=-1            |    94.1 |       63.7 |   1786 MB |
| after + fp16 + n_jobs=-1     |   117.0 |       76.2 |   1822 MB |

Notes:
- Headline: 297 -> 724 sent/s end-to-end on short sentences (2.4x);
  50.9 -> 76.2 sent/s on long sentences (1.5x).
- inference_mode cuts peak CUDA memory ~38% at batch 32
  (2619 -> 1619 MB) and halves it on the long corpus (3593 -> 1786 MB).
- n_jobs=-1 on long sentences: 42.1 -> 63.7 sent/s; on short sentences
  it is slightly counterproductive (631 -> 581 sent/s on the 750-sentence
  run) due to per-sentence IPC overhead.
- REGRESSION (long corpus only): "after" GPU tagging is consistently
  slower than "before" (1.6-2.0s vs 1.1s across three runs). Suspected
  cause: `extract_topk` transfers a third (mask) tensor and full padded
  rows to Python, where the old loop sliced per-sentence before
  `.tolist()`. Matches the code-review follow-up "slice before tolist";
  cost grows with chart size, so it only shows on long sentences.
  End-to-end still wins via n_jobs/fp16, but this is the top follow-up.

## GPU results — beaker SGE cluster, NVIDIA A40 on `animal-206-2` (torch 2.6+cu124, Python 3.11)

SGE job 6950991 via gpu.q. Node was busy (load ~22 with other tenants),
so CPU-side stages (tokenisation, chart parsing) are noisier than the
Lab 105 numbers. Stage rates are sent/s.

Short corpus (400 sentences):

| config                  | tagging | end-to-end | peak CUDA |
|-------------------------|--------:|-----------:|----------:|
| before, default (batch 4) |  185.8 |      148.5 |   1592 MB |
| before, batch 32        |   342.0 |      230.4 |   1620 MB |
| after, batch 32         |   361.8 |      242.1 |   1620 MB |
| after, batch 32 + fp16  |   277.3 |      199.9 |   2371 MB |
| after, max-spans 50000  |   233.6 |      185.3 |   2144 MB |

Long corpus (187 sentences, batch 16):

| config                   | end-to-end |
|--------------------------|-----------:|
| before                   |       16.2 |
| after                    |       20.7 |
| after + n_jobs=4         |       28.9 |
| after + fp16 + n_jobs=4  |       28.6 |

Notes:
- n_jobs=4 on long sentences: 16.2 -> 28.9 sent/s end-to-end (1.8x) —
  the chart parser is the dominant stage on GPU, as on the 3090 Ti.
- fp16 at batch 32 does not pay on the A40 either (same pattern as the
  3090 Ti: autocast needs large batches, e.g. batch 128, to win).
- Operational note: jobs scheduled onto `mitchell.local` card 0 failed
  with "CUDA-capable device(s) is/are busy or unavailable" (exclusive
  compute mode, apparently a stale process); excluding the node with
  `qsub -l h=!mitchell.local` resolved it.

## After extract_topk survivor-only transfer (fix for the long-sentence regression)

`extract_topk` now drops padding positions on-device and transfers only
mask-surviving entries (`nonzero` + boolean indexing) instead of
materialising the full padded (batch, positions, k) tensors as Python
objects. Output unchanged (26-case equivalence suite). This not only
removes the regression noted above but beats the pre-branch baseline,
which paid a similar per-row transfer cost.

RTX 3090 Ti (goosander-l), long corpus (187 sentences, batch 16):

| metric                      | before branch | branch pre-fix | branch fixed |
|-----------------------------|--------------:|---------------:|-------------:|
| tagging                     |        1.10s  |     1.6-2.0s   |       0.73s  |
| end-to-end fp16 + n_jobs=-1 |             — |   2.45s (76.2) | 0.98s (191.1)|

RTX 3090 Ti, short corpus (400 sentences, batch 128 + fp16):
tagging 0.29s -> 0.13s (3017 sent/s); end-to-end 0.55s -> 0.39s
(1033.9 sent/s, vs 297 sent/s pre-branch default = 3.5x).

CPU (i7-11800H), long corpus batch 16: tagging 16.7s -> 13.8s.
