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
