# BobcatParser throughput benchmark results

Corpus: 750 generated sentences (3-12 words), `/tmp/bobcat_bench.txt`,
see Task 1 of `docs/superpowers/plans/2026-06-10-bobcat-throughput.md`.
Machine: Intel Core i7-11800H @ 2.30GHz, GPU: RTX 3080 Laptop present but CUDA unavailable (driver mismatch); GPU numbers to be collected later, RAM: 31 GB.

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
