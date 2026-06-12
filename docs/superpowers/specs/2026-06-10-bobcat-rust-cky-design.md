# bobcat_rs: Rust CKY core for BobcatParser — design

**Date:** 2026-06-10
**Status:** approved
**Predecessor:** `2026-06-10-bobcat-throughput-design.md` (Tier 1, landed on
branch `bobcat-throughput`)

## Background and evidence

After Tier 1, the pure-Python chart parser is the dominant cost everywhere
that matters:

- RTX 3090 Ti, long corpus (187 sentences, 25-50 words): tagging 0.73s vs
  chart parsing 2.5s serial.
- Short corpus at batch 128 + fp16: tagging 0.13s vs chart parsing 0.25s.
- cProfile (60 long sentences, 3.7s total): 7.9M Python function calls;
  `rules.combine` x326k, category `_equals` x703k, `matches` x249k. No
  single hot leaf — the cost is interpreter overhead across the whole
  inner machinery. Python-level optimization ceiling is ~1.5-2x.

Decision (with user): full-stage Rust port (PyO3), not leaf offloading
(boundary crossings would cap gains at 2-3x) and not class-replacement
(keeps Python loop overhead, breaks pickling).

## Goals & non-goals

- **Goals:** >=10x single-thread speedup of the chart-parse stage;
  rayon batch parallelism across sentences (GIL released); identical
  best-parse trees to the Python implementation; Python implementation
  retained as oracle and fallback.
- **Non-goals:** dependency output (`deps_and_tags`) in Rust; tagger
  changes; upstreaming (this is fork-tier work on `bobcat-throughput`);
  publishing wheels (build from source per machine for now).

## Architecture

```
lambeq/bobcat/parser.py   ChartParser (Python, unchanged logic)
        |                       ^ fallback / oracle
        v                       |
lambeq/bobcat/rust_backend.py --> backend='auto'|'rust'|'python'
        |
        v
rust/  (maturin crate "bobcat_rs", PyO3 >= 0.22)
  src/category.rs   markedup category parsing, equality, matching, vars
  src/grammar.rs    grammar.json model: binary rules, type-changing,
                    type-raising, result-cat table
  src/rules.rs      combinators + Eisner NF constraints + coordination
  src/chart.rs      Cell (beam, per-category dedup), Chart
  src/parser.rs     CKY loop, scoring, root filtering, parse_batch (rayon)
  src/lib.rs        PyO3 bindings: RustChartParser class
```

### Python-side integration

- New module `lambeq/bobcat/rust_backend.py`: wraps `bobcat_rs` behind the
  same call contract as `ChartParser.__call__` (Sentence in, object whose
  `[0]` is the best `ParseTree`, falsy when no parse).
- `BobcatParser` gains kwarg `parser_backend: str = 'auto'`:
  - `'auto'`: use Rust if `import bobcat_rs` succeeds, else Python.
  - `'rust'`: require it; `ImportError` with build instructions otherwise.
  - `'python'`: force the oracle.
- `n_jobs` (Tier 1) remains valid for the Python backend; when the Rust
  backend is active, batch parallelism happens inside `parse_batch` and
  `n_jobs` is ignored with a warning.
- `sentences2trees` calls `parse_batch` once per tagged batch when the
  Rust backend is active (instead of the per-sentence loop).

### The boundary

One-time construction (per BobcatParser): plain-Python dicts/lists derived
from `grammar.json`, the model's category list, and the existing
ChartParser config (eisner_normal_form, max_parse_trees, beam_size,
input_tag_score_weight, missing_cat_score, missing_span_score, root_cats).

Per batch: `parse_batch(sentences)` where each sentence is
`(words: list[str], supertags: list[list[tuple[str, float]]],
span_scores: dict[tuple[int, int], dict[int, float]])` — the same data
`_prepare_sentence` builds today.

Return per sentence: `None` (no parse / failure detail via exception) or a
flat node list, each node `(rule_name: str, cat: str (markedup), word:
str | None, left: int, right: int)` with -1 for absent children, root
last. Python rebuilds a genuine `ParseTree` for the best derivation only
(~2n nodes) so `_build_ccgtree` and `metadata['original']` work
unchanged. The rebuilt tree's dependency fields are left unfilled — they
do not affect tree selection (scores are tag + span scores only) and
nothing in the diagram pipeline reads them.

## Fidelity requirements (identical-trees bar)

The Rust core must replicate, exactly:

1. **Category semantics** (`lexicon.py`): markedup parsing including
   variable slots, `Category.__eq__`/`_equals`, `matches`, `translate`,
   NP-feature handling (`cat.atom == Atom.NP` fallback in scoring).
2. **Rule application** (`rules.py`): every combinator's preconditions,
   Eisner normal-form constraints, coordination flagging, type-raising
   and type-changing tables, `match_rule` semantics.
3. **Beam semantics** (`parser.py` `Cell.add`): descending-score sorted
   insertion with binary search, one-tree-per-category replacement rule,
   beam cutoff including the keep-ties behaviour, `min_score` short
   circuit, `parse_tree_count`/`max_parse_trees` cutoff.
4. **Scoring**: float additions in the same order as `calc_score_unary`/
   `calc_score_binary`/`get_span_score` (f64 throughout, like CPython).
5. **Tie-breaking**: where Python relies on stable sort order or
   insertion order, Rust must reproduce it (stable sorts, same iteration
   order over candidates).

Known acceptable divergence: none at the tree level. If corpus testing
surfaces an unavoidable tie ambiguity, it is resolved by matching the
Python behaviour, not by relaxing the test.

## Testing

1. **Unit differential tests** (pytest, skipped when `bobcat_rs` absent):
   - category parse/equality/matching: Rust vs Python over all categories
     in the real grammar plus pairwise samples;
   - single rule applications: for sampled category pairs, Rust
     `combine` output (rule, result category) equals Python
     `Rules.combine`.
2. **Corpus equivalence**: all 937 benchmark sentences (short + long
   corpora) parse to trees where `CCGTree` built via Rust == via Python
   (uses `CCGTree.__eq__`). Any mismatch is a test failure.
3. **Existing suite**: the 62 Bobcat tests pass with `parser_backend`
   forced to each of `'python'` and `'rust'`.
4. **Benchmarks**: extend `benchmarks/bobcat_throughput.py` with
   `--parser-backend`; record before/after on laptop CPU, goosander
   (3090 Ti), beaker (A40).

## Error handling

- Rust panics are caught at the PyO3 boundary and surface as Python
  exceptions; `sentences2trees` wraps them in `BobcatParseError` exactly
  like Python-backend failures, respecting `suppress_exceptions`.
- `max_parse_trees` exceeded behaves as in Python (parse continues to
  yield best-so-far result; same chart cutoff).
- `parser_backend='rust'` without the extension: `ImportError` naming
  `pip install ./rust` (maturin) as the fix.
- Invalid `parser_backend` value: `ValueError` at construction.

## Build & environments

- `rust/pyproject.toml` with maturin; `pip install ./rust` builds the
  extension into the active env. Rust toolchain via rustup.
- Must build on: the laptop (qnlp env), Lab 105 pool (glibc 2.34+), and
  beaker (CentOS 7, glibc 2.17 — rustup works there; PyO3 abi3 wheel
  built on the host itself).
- No prebuilt wheels for now.

## Performance targets

- Single-thread: chart stage on the long corpus (187 sentences)
  2.5s -> <=0.25s.
- `parse_batch` with rayon on >=8 cores: chart stage <=0.1s on the same
  corpus, i.e. tagging becomes the bottleneck again on GPU.
- If single-thread lands short of 5x, stop and re-profile before adding
  rayon.

## Delivery phases

1. Crate scaffold + category module + differential tests.
2. Grammar + rules + differential tests.
3. Chart/beam + single-sentence CKY + corpus equivalence (serial).
4. `parse_batch` + rayon + Python dispatch/integration + existing suite
   on both backends + benchmarks on the three machines.

Each phase is independently testable; phases 1-3 produce no user-visible
change until the dispatch lands in phase 4.
