# BobcatParser throughput benchmark results

Corpus: 750 generated sentences (3-12 words), `/tmp/bobcat_bench.txt`,
see Task 1 of `docs/superpowers/plans/2026-06-10-bobcat-throughput.md`.
Machine: Intel Core i7-11800H @ 2.30GHz, GPU: RTX 3080 Laptop present but CUDA unavailable (driver mismatch); GPU numbers to be collected later, RAM: 31 GB.

## Baseline (before optimizations)

```
config:           {} {} device=cpu
sentences:        200 (0 failed)
tagging:          8.19s (24.4 sent/s)
chart parsing:    0.24s (842.5 sent/s)
end-to-end:       8.43s (23.7 sent/s)
peak RSS:         2650 MB
```
