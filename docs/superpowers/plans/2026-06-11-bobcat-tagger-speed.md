# Tier 3 Faster Supertagger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift GPU tagging from ~300 to >=1000 sent/s on long sentences via tuned defaults, a raw topk->Rust handoff that eliminates Python post-processing, and opt-in torch.compile / ONNX Runtime lanes.

**Architecture:** The tagger keeps its classic `__call__`/`TaggerOutput` lane (Python backend, external users). A new fused lane — `Tagger.forward_topk` producing CPU topk tensors, consumed zero-copy by `bobcat_rs.parse_batch_raw` which replicates `extract_topk` + `_prepare_sentence` semantics in Rust — is used by `sentences2trees` whenever the Rust parser backend is active. GPU defaults (fp16, batch 64) switch on automatically for CUDA devices unless explicitly overridden.

**Tech Stack:** PyTorch, PyO3 + `numpy` crate (zero-copy buffers), rayon, torch.compile, onnxruntime (optional), pytest differential gates.

**Spec:** `docs/superpowers/specs/2026-06-11-bobcat-tagger-speed-design.md`

---

## Conventions (every task)

- Repo `/home/kinianlo/projects/lambeq`, branch `bobcat-throughput`.
  `PY=~/.pyenv/versions/qnlp/bin/python`. Rebuild extension:
  `PATH=$PATH:~/.cargo/bin $PY -m pip install -q ./rust`.
- 600000ms timeouts on builds and on any pytest touching the
  `bobcat_parser` fixture (loads a 1.4GB BERT).
- Oracles win over prompt summaries: `lambeq/bobcat/tagger.py`
  (`extract_topk`, `chart_size`, `get_chart_spans`/`SPAN_MEMO`),
  `lambeq/text2diagram/model_based_reader/bobcat_parser.py`
  (`_prepare_sentence`, the rust dispatch), `rust/src/*.rs`.
- CUDA is NOT available on this laptop — GPU verification happens in
  Task 5 on goosander/beaker. All local tests are CPU.
- Commit style: plain sentence + trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File map

```
lambeq/text2diagram/model_based_reader/bobcat_parser.py
    _apply_gpu_tagger_defaults (new helper), fused-lane dispatch,
    compile_model + tagger_backend kwargs
lambeq/bobcat/tagger.py
    Tagger._model_output refactor, forward_topk, onnx forward path,
    BertForChartClassification.classify refactor
lambeq/bobcat/rust_backend.py     RustBackend.configure_raw / parse_raw
rust/Cargo.toml                   + numpy = "0.22"
rust/src/parser.rs                raw thresholding/assembly + parse_batch_raw
rust/src/lib.rs                   PyO3: configure_raw, parse_batch_raw
tools/export_onnx.py              ONNX body export script
benchmarks/bobcat_throughput.py   --compile-model, --tagger-backend
tests/test_bobcat_rs.py           raw-lane differential + equivalence tests
tests/text2diagram/model_based_reader/test_bobcat_parser.py
                                  defaults + compile + onnx tests
```

---

### Task 1: Tuned GPU defaults

**Files:**
- Modify: `lambeq/text2diagram/model_based_reader/bobcat_parser.py`
- Test: `tests/text2diagram/model_based_reader/test_bobcat_parser.py`

- [ ] **Step 1: Write the failing tests** (append to the parser test file):

```python
def test_gpu_tagger_defaults_helper():
    from lambeq.text2diagram.model_based_reader.bobcat_parser import (
        _apply_gpu_tagger_defaults)
    # CUDA, nothing user-set, shipped config: both defaults apply
    config = {'batch_size': 4}
    _apply_gpu_tagger_defaults(config, 'cuda', set())
    assert config == {'batch_size': 64, 'dtype': 'float16'}
    # explicit user settings always win
    config = {'batch_size': 4, 'dtype': 'bfloat16'}
    _apply_gpu_tagger_defaults(config, 'cuda', {'batch_size', 'dtype'})
    assert config == {'batch_size': 4, 'dtype': 'bfloat16'}
    # non-shipped pipeline batch_size is respected
    config = {'batch_size': 32}
    _apply_gpu_tagger_defaults(config, 'cuda', set())
    assert config['batch_size'] == 32 and config['dtype'] == 'float16'
    # CPU: untouched
    config = {'batch_size': 4}
    _apply_gpu_tagger_defaults(config, 'cpu', set())
    assert config == {'batch_size': 4}


def test_cpu_parser_keeps_classic_defaults(bobcat_parser):
    assert bobcat_parser.tagger.dtype is None
    assert bobcat_parser.tagger.batch_size == 4
```

- [ ] **Step 2: Verify failure** — `$PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py::test_gpu_tagger_defaults_helper -q`: ImportError on `_apply_gpu_tagger_defaults`.

- [ ] **Step 3: Implement.** In `bobcat_parser.py`, module level (near the worker functions):

```python
def _apply_gpu_tagger_defaults(tagger_config: dict,
                               device_type: str,
                               user_set: set[str]) -> None:
    """Default to fp16 + batch 64 on CUDA devices.

    Applied only when the user did not set the key explicitly and the
    pipeline config is at its shipped value ('dtype' absent;
    batch_size == 4).
    """
    if device_type != 'cuda':
        return
    if 'dtype' not in user_set and tagger_config.get('dtype') is None:
        tagger_config['dtype'] = 'float16'
    if 'batch_size' not in user_set and tagger_config.get('batch_size') == 4:
        tagger_config['batch_size'] = 64
```

In `_initialise_model`, capture at the very top (before the config
merge loop): `user_set = {k for k in ('dtype', 'batch_size') if k in kwargs}`.
After the config dict is fully assembled (after the postdates-keys
loop), and before constructing the `Tagger`:

```python
        _apply_gpu_tagger_defaults(config['tagger'],
                                   torch.device(self.device).type,
                                   user_set)
```

Add to the `BobcatParser.__init__` docstring (device parameter notes or
Other Parameters intro):

```
            On CUDA devices the tagger defaults change to
            `dtype='float16'` and `batch_size=64` unless these are
            passed explicitly.
```

- [ ] **Step 4: Run** the parser test file (`-q`, expect 39 passed) and `tests/test_bobcat.py` (27).

- [ ] **Step 5: Commit**

```bash
git add lambeq/text2diagram/model_based_reader/bobcat_parser.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py
git commit -m "Default Bobcat tagger to fp16 and batch 64 on CUDA

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Raw topk handoff (core)

**Files:**
- Modify: `lambeq/bobcat/tagger.py`, `lambeq/bobcat/rust_backend.py`,
  `lambeq/text2diagram/model_based_reader/bobcat_parser.py`,
  `rust/Cargo.toml`, `rust/src/parser.rs`, `rust/src/lib.rs`
- Test: `tests/test_bobcat_rs.py`

#### Contracts

**`Tagger._model_output(encodings)`** — refactor: extract the existing
autocast/nullcontext + `self.model(...)` block from `parse` into a
private method used by both `parse` and `forward_topk` (no behaviour
change; `parse` keeps its `@torch.inference_mode()`).

**`Tagger.forward_topk(inputs)`**:

```python
    @torch.inference_mode()
    def forward_topk(
        self,
        inputs: Sequence[Sequence[str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the model and return CPU top-k tensors for the raw lane.

        Returns (tag_scores [B, W, k_tag] f32, tag_indices i64,
        span_scores [B, S, k_span] f32, span_indices i64), where W/S are
        the padded word/span counts. Thresholding happens downstream
        (bobcat_rs), with semantics identical to `extract_topk`.
        """
        encodings = self.prepare_inputs(inputs, word_mask=True)
        outputs = self._model_output(encodings)

        def topk(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor,
                                                            torch.Tensor]:
            k = min(top_k, logits.size(-1)) if top_k else logits.size(-1)
            scores, indices = logits.float().log_softmax(-1).topk(k)
            return scores.cpu().contiguous(), indices.cpu().contiguous()

        tag_scores, tag_indices = topk(outputs.tag_logits, self.tag_top_k)
        span_scores, span_indices = topk(outputs.span_logits,
                                         self.span_top_k)
        return tag_scores, tag_indices, span_scores, span_indices
```

(`logits.float().log_softmax(-1).topk(k)` — the exact op order of
`extract_topk`, so scores are bit-identical.)

**Rust side.** Add `numpy = "0.22"` to `rust/Cargo.toml`. In
`rust/src/parser.rs` (or a new `raw.rs` if cleaner):

- `RawConfig { tags: Vec<Option<CatRef>>, tag_prob_threshold: f64,
  tag_strategy_relative: bool, span_prob_threshold: f64,
  span_strategy_relative: bool, input is already weighted downstream }`
  — `tags[i]` is the markedup category for model tag id `i`, resolved
  through the grammar's plain->markedup `categories` map at configure
  time (`None` when the tag string is not a lexical category — a
  sentence using it fails like the classic lane's missing-key path).
- `parse_one_raw(words, tag_scores 2D view for this sentence, ...)`
  assembles the SAME internal inputs `parse_one` consumes:
  - per word w (w < words.len()): iterate k descending; threshold
    exactly like `extract_topk`: `prob_threshold == 0` -> keep all;
    relative -> `score >= row_top + ln(p) as f32` computed in f32
    (torch adds the f64 scalar after casting to f32 — match that:
    `let thr = row0 + (p.ln() as f32)`); absolute -> `score >= (p.ln()
    as f32)`. Scores descending => break at first below-threshold.
    Tag index 0 is KEPT (only spans skip 0). Supertag prob = score
    widened `as f64` (same as Python float()-of-f32).
  - per span position s (s < chart_size(words.len())): same threshold
    with the span params; skip entries with index 0 but CONTINUE the
    scan (prefix-break only on threshold). Span (start, end) from
    position s follows `get_chart_spans` ordering (tagger.py:50-63):
    `for end in 0..W { for start in (0..=end).rev() { ... } }`.
    cat_id = index `as u32`, score `as f64`.
  - then run the existing CKY exactly as `parse_one` does (factor the
    common core so the classic `parse_one` and `parse_one_raw` share
    everything after input assembly).
- PyO3 (`lib.rs`):

```rust
fn configure_raw(&mut self, tags: Vec<String>,
                 tag_prob_threshold: f64, tag_prob_threshold_strategy: String,
                 span_prob_threshold: f64, span_prob_threshold_strategy: String)
                 -> PyResult<()>;
#[pyo3(signature = (words, tag_scores, tag_indices, span_scores, span_indices, num_threads = 0))]
fn parse_batch_raw(&self, py: Python<'_>, words: Vec<Vec<String>>,
                   tag_scores: numpy::PyReadonlyArray3<'_, f32>,
                   tag_indices: numpy::PyReadonlyArray3<'_, i64>,
                   span_scores: numpy::PyReadonlyArray3<'_, f32>,
                   span_indices: numpy::PyReadonlyArray3<'_, i64>,
                   num_threads: usize)
                   -> PyResult<Vec<Option<Vec<SerNode>>>>;
```

  - validate shapes: dim0 of all four == words.len(); W dim >= max word
    count; S dim >= chart_size(max word count); raise PyValueError on
    mismatch (not panic). `parse_batch_raw` without `configure_raw`
    -> PyValueError. Copy the arrays' relevant slices into owned Rust
    buffers (or use `as_array()` views — note `allow_threads` cannot
    borrow the Python-held arrays, so copy into `Vec<f32>`/`Vec<i64>`
    BEFORE releasing the GIL; the copy is a memcpy, negligible next to
    55ms of Python loops). Same rayon/num_threads semantics as
    `parse_batch`.

**`RustBackend`** (`rust_backend.py`) — add:

```python
    def configure_raw(self, tags, tag_prob_threshold,
                      tag_prob_threshold_strategy, span_prob_threshold,
                      span_prob_threshold_strategy) -> None:
        self._parser.configure_raw(list(tags),
                                   tag_prob_threshold,
                                   tag_prob_threshold_strategy,
                                   span_prob_threshold,
                                   span_prob_threshold_strategy)
        self.raw_configured = True

    def parse_raw(self, words, tag_scores, tag_indices, span_scores,
                  span_indices, num_threads: int = 0):
        results = self._parser.parse_batch_raw(
            [list(w) for w in words],
            tag_scores.numpy(), tag_indices.numpy(),
            span_scores.numpy(), span_indices.numpy(),
            num_threads)
        return [nodes_to_tree(nodes) if nodes is not None else None
                for nodes in results]
```

(set `self.raw_configured = False` in `__init__`.)

**`BobcatParser`** — in `_initialise_model`, after constructing the
rust backend AND the tagger (order: tagger exists first), call:

```python
            self.parser.configure_raw(
                self.tagger.model.config.tags,
                self.tagger.tag_prob_threshold,
                self.tagger.tag_prob_threshold_strategy,
                self.tagger.span_prob_threshold,
                self.tagger.span_prob_threshold_strategy)
```

NOTE the classic lane applies `input_tag_score_weight` inside
`parse_one` lexical scoring — already shared by the refactored core, so
the raw lane must hand over RAW log-probs (it does).

In `sentences2trees`, replace the rust branch's body (tagging happens
OUTSIDE the branch today — restructure so the fused lane skips classic
tagging entirely, while keeping the TEXT-verbosity prints, which an
existing test asserts verbatim):

```python
        trees: list[CCGTree] = []
        if sentences_valid:
            if verbose == VerbosityLevel.TEXT.value:
                print('Tagging sentences.', file=sys.stderr)
            if getattr(self, 'parser_backend', 'python') == 'rust' \
                    and isinstance(self.parser, RustBackend):
                if n_jobs != 1:
                    warnings.warn('`n_jobs` is ignored with the Rust '
                                  'parser backend; it parallelises '
                                  'internally', stacklevel=2)
                if verbose == VerbosityLevel.TEXT.value:
                    print('Parsing tagged sentences.', file=sys.stderr)
                trees = self._parse_fused(sentences_valid,
                                          suppress_exceptions, verbose)
            else:
                tag_results = self.tagger(sentences_valid, verbose=verbose)
                tags = tag_results.tags
                if verbose == VerbosityLevel.TEXT.value:
                    print('Parsing tagged sentences.', file=sys.stderr)
                if n_jobs == 1:
                    ... existing serial loop, unchanged ...
                else:
                    ... existing multiprocessing branch, unchanged ...
```

and add the method:

```python
    def _parse_fused(self,
                     sentences: list[list[str]],
                     suppress_exceptions: bool,
                     verbose: str) -> list[CCGTree]:
        """Tag and parse via the fused raw-tensor Rust lane."""
        tagger = self.tagger
        trees: list[CCGTree | None] = [None] * len(sentences)
        batches = tagger.make_batches(sentences, tagger.batch_size)
        for batch in tqdm(
                batches,
                desc='Parsing sentences',
                leave=False,
                disable=verbose != VerbosityLevel.PROGRESS.value):
            words = [list(sentences[i]) for i in batch]
            raw = tagger.forward_topk(words)
            try:
                parse_trees = self.parser.parse_raw(words, *raw)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                # Rust panics derive from BaseException, not Exception
                if suppress_exceptions:
                    parse_trees = [None] * len(batch)
                else:
                    raise BobcatParseError(
                        f'<Rust backend batch of {len(batch)} '
                        f'sentences>') from e
            for i, parse_tree in zip(batch, parse_trees):
                if parse_tree is not None:
                    trees[i] = self._build_ccgtree(parse_tree)
                elif not suppress_exceptions:
                    raise BobcatParseError(' '.join(sentences[i]))
        return trees
```

(The old rust-branch code that built `sentence_inputs` via
`_prepare_sentence` is REMOVED from `sentences2trees`; `parse_batch`
stays on `RustBackend` for direct users and tests.)

#### Steps

- [ ] **Step 1: failing tests** — append to `tests/test_bobcat_rs.py`:

```python
def _raw_configured_parser(grammar_data, bobcat_parser):
    rs = bobcat_rs.RustChartParser(
        grammar_data['categories'],
        [tuple(r) for r in grammar_data['binary_rules']],
        [tuple(r) for r in grammar_data['type_changing_rules']],
        [tuple(r) for r in grammar_data['type_raising_rules']],
        bobcat_parser.tagger.model.config.cats,
        None, True, 50000, 32, 1.0, 0.01, 1e-05)
    t = bobcat_parser.tagger
    rs.configure_raw(t.model.config.tags,
                     t.tag_prob_threshold, t.tag_prob_threshold_strategy,
                     t.span_prob_threshold, t.span_prob_threshold_strategy)
    return rs


def test_parse_batch_raw_matches_classic_lane(grammar_data, bobcat_parser,
                                              rs_parser):
    rs_raw = _raw_configured_parser(grammar_data, bobcat_parser)
    out = _tagged(bobcat_parser, SENTENCES)
    classic = rs_parser.parse_batch(_rust_inputs(bobcat_parser, out))
    words = [s.split() for s in SENTENCES]
    raw = bobcat_parser.tagger.forward_topk(words)
    raw_results = rs_raw.parse_batch_raw(
        words, raw[0].numpy(), raw[1].numpy(),
        raw[2].numpy(), raw[3].numpy())
    assert raw_results == classic


def test_parse_batch_raw_requires_configuration(grammar_data,
                                                bobcat_parser, rs_parser):
    import numpy as np
    z3f = np.zeros((1, 1, 1), dtype=np.float32)
    z3i = np.zeros((1, 1, 1), dtype=np.int64)
    with pytest.raises(ValueError):
        rs_parser.parse_batch_raw([['a']], z3f, z3i, z3f, z3i)


def test_fused_lane_in_sentences2trees(bobcat_parser):
    from lambeq import BobcatParser, VerbosityLevel
    rust_parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                               parser_backend='rust')
    assert (rust_parser.sentences2trees(
                SENTENCES, verbose=VerbosityLevel.SUPPRESS.value)
            == bobcat_parser.sentences2trees(
                SENTENCES, verbose=VerbosityLevel.SUPPRESS.value))


@pytest.mark.parametrize('tag_p,tag_strat,span_p,span_strat', [
    (0.002, 'relative', 0.0003, 'relative'),   # shipped config
    (0.01, 'absolute', 0.001, 'absolute'),
    (0, 'relative', 0, 'relative'),            # keep-all
    (1, 'relative', 1, 'relative'),            # top-score-only
])
def test_raw_thresholding_matches_extract_topk(grammar_data, bobcat_parser,
                                               tag_p, tag_strat,
                                               span_p, span_strat):
    """Random logits, all threshold strategies: classic assembly
    (extract_topk + Sentence tuples -> parse_batch) must equal the raw
    lane (topk tensors -> parse_batch_raw) tree-for-tree."""
    import torch
    from lambeq.bobcat.tagger import chart_size, extract_topk, idx2span
    tagger = bobcat_parser.tagger
    n_tags = len(tagger.model.config.tags)
    n_cats = len(tagger.model.config.cats)
    words = [['w%d' % j for j in range(n)] for n in (3, 7, 12)]
    W = max(len(w) for w in words)
    S = chart_size(W)
    torch.manual_seed(42)
    tag_logits = torch.randn(len(words), W, n_tags)
    span_logits = torch.randn(len(words), S, n_cats)

    rs = bobcat_rs.RustChartParser(
        grammar_data['categories'],
        [tuple(r) for r in grammar_data['binary_rules']],
        [tuple(r) for r in grammar_data['type_changing_rules']],
        [tuple(r) for r in grammar_data['type_raising_rules']],
        bobcat_parser.tagger.model.config.cats,
        None, True, 50000, 32, 1.0, 0.01, 1e-05)
    rs.configure_raw(tagger.model.config.tags,
                     tag_p, tag_strat, span_p, span_strat)

    # classic assembly (mirrors Tagger.parse + _prepare_sentence)
    tags_list = tagger.model.config.tags
    tag_out = extract_topk(tag_logits, [len(w) for w in words],
                           tagger.tag_top_k, tag_p, tag_strat, False)
    span_out = extract_topk(span_logits,
                            [chart_size(len(w)) for w in words],
                            tagger.span_top_k, span_p, span_strat, True)
    classic_inputs = []
    for w, t_rows, s_rows in zip(words, tag_out, span_out):
        supertags = [[(tags_list[i], sc) for i, sc in row] for row in t_rows]
        spans = {idx2span(i): {ci: cs for ci, cs in row}
                 for i, row in enumerate(s_rows) if row}
        classic_inputs.append((w, supertags, spans))
    classic = rs.parse_batch(classic_inputs)

    def topk(logits, k):
        s, i = logits.float().log_softmax(-1).topk(k)
        return s.cpu().contiguous(), i.cpu().contiguous()

    ts, ti = topk(tag_logits, tagger.tag_top_k)
    ss, si = topk(span_logits, tagger.span_top_k)
    raw = rs.parse_batch_raw(words, ts.numpy(), ti.numpy(),
                             ss.numpy(), si.numpy())
    assert raw == classic
```

**ALSO in this step:** update the existing `bobcat_parser` fixture in
`tests/test_bobcat_rs.py` to construct
`BobcatParser(verbose=..., parser_backend='python')` explicitly — it is
the PYTHON oracle for these tests (used for tagger access and
`bobcat_parser.parser(...)` oracle calls), and with the extension built
the unpinned constructor would now resolve to rust.

- [ ] **Step 2: verify failure** — run the new tests: AttributeError
  (`configure_raw`). Existing 11 must still pass.

- [ ] **Step 3: implement** per the contracts (Rust first, rebuild,
  then tagger/rust_backend/bobcat_parser). Iterate to green:

```bash
PATH=$PATH:~/.cargo/bin $PY -m pip install -q ./rust
$PY -m pytest tests/test_bobcat_rs.py -q          # 18 passed
```

- [ ] **Step 4: full gauntlet**

```bash
LAMBEQ_BOBCAT_BACKEND=python $PY -m pytest tests/test_bobcat.py tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
LAMBEQ_BOBCAT_BACKEND=rust   $PY -m pytest tests/test_bobcat.py tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all green both runs (the rust run now exercises the fused
lane through every sentences2trees-based test, including the verbatim
TEXT-verbosity progress test and root-cat filtering).

- [ ] **Step 5: corpus gate via the fused lane**

```bash
$PY benchmarks/check_equivalence.py /tmp/bobcat_bench.txt /tmp/bobcat_bench_long.txt
```

Expected: `937/937 identical` (the rust side now runs fused
automatically). Regenerate corpora per Tier-1 plan Task 1/7 if /tmp was
wiped. ANY mismatch: debug Rust thresholding (off-by-one in span
ordering and the f32 threshold arithmetic are the likely culprits); do
not weaken the gate.

- [ ] **Step 6: commit**

```bash
git add rust/ lambeq/ tests/test_bobcat_rs.py
git commit -m "Add raw top-k handoff lane from tagger to Rust parser

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: torch.compile flag

**Files:**
- Modify: `lambeq/text2diagram/model_based_reader/bobcat_parser.py`
- Test: `tests/text2diagram/model_based_reader/test_bobcat_parser.py`

- [ ] **Step 1: failing test** (append):

```python
def test_compile_model_flag(bobcat_parser):
    with pytest.raises(ValueError):
        BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                     compile_model='yes')
    compiled = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                            compile_model=True)
    assert type(compiled.tagger.model.bert).__name__ == 'OptimizedModule'
```

- [ ] **Step 2: implement.** `BobcatParser.__init__` gains
`compile_model: bool = False` (docstring: wraps the BERT encoder with
`torch.compile(dynamic=True)`; first call pays compilation latency;
worth it for corpus-scale runs), passed to `_initialise_model`, which
validates (`if not isinstance(compile_model, bool): raise ValueError`)
and, after the model is loaded and moved to device:

```python
        if compile_model:
            model.bert = torch.compile(model.bert, dynamic=True)
```

(no forward in the test - wrapping is lazy and cheap; numeric
verification happens in the Task 5 GPU runs through the corpus
checker.)

- [ ] **Step 3: run + commit**

```bash
$PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
git add lambeq/text2diagram/model_based_reader/bobcat_parser.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py
git commit -m "Add opt-in torch.compile for the Bobcat tagger encoder

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: ONNX Runtime lane

**Files:**
- Create: `tools/export_onnx.py`
- Modify: `lambeq/bobcat/tagger.py` (classify refactor + onnx forward),
  `lambeq/text2diagram/model_based_reader/bobcat_parser.py`
  (`tagger_backend` kwarg)
- Test: `tests/text2diagram/model_based_reader/test_bobcat_parser.py`

Decision locked from the spec's allowed fallback: export the **BERT
body only** (the word-mask gather in the heads is data-dependent and
not worth fighting the exporter for 6.6ms of head time); heads stay in
torch via a `classify` refactor.

- [ ] **Step 1: refactor `BertForChartClassification`.** Extract the
post-BERT section of `forward` (from `sequence_output = outputs[0]`
through `span_logits = ...`) into:

```python
    def classify(
        self,
        sequence_output: torch.Tensor,
        word_mask: torch.BoolTensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the tag/span heads on encoder output."""
        ... (moved code, returning (tag_logits, span_logits)) ...
```

`forward` calls it. Run the full bobcat suites (both backends) to prove
no behaviour change; commit separately:

```bash
git add lambeq/bobcat/tagger.py
git commit -m "Factor classification heads out of BertForChartClassification.forward

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: export script.** `tools/export_onnx.py`:

```python
#!/usr/bin/env python3
"""Export the Bobcat BERT encoder to ONNX (body only; heads run in torch).

Usage: python tools/export_onnx.py [model_dir]
Default model_dir: ~/.cache/lambeq/bobcat/bobcat
Writes <model_dir>/bobcat-body.onnx
"""
import sys
from pathlib import Path

import torch

from lambeq.bobcat import BertForChartClassification


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path.home() / '.cache/lambeq/bobcat/bobcat')
    model = BertForChartClassification.from_pretrained(model_dir).eval()
    dummy = {
        'input_ids': torch.ones(2, 8, dtype=torch.long),
        'attention_mask': torch.ones(2, 8, dtype=torch.long),
        'token_type_ids': torch.zeros(2, 8, dtype=torch.long),
    }
    out_path = model_dir / 'bobcat-body.onnx'
    torch.onnx.export(
        model.bert,
        (dummy['input_ids'], dummy['attention_mask'],
         dummy['token_type_ids']),
        str(out_path),
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={name: {0: 'batch', 1: 'sequence'}
                      for name in ('input_ids', 'attention_mask',
                                   'token_type_ids',
                                   'last_hidden_state')},
        opset_version=17)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
```

NOTE: `model.bert(...)` returns a ModelOutput; if the exporter
complains, wrap with a tiny `nn.Module` whose forward returns
`self.bert(...)[0]` — implement that wrapper inside the script if
needed (decide by running it).

- [ ] **Step 3: tagger ONNX path.** `Tagger.__init__` gains
`tagger_backend: str = 'torch'` and `onnx_session=None` (an initialised
`onnxruntime.InferenceSession` injected by `_initialise_model`;
validation: `tagger_backend in ('torch', 'onnx')` else ValueError;
`'onnx'` requires a session). `_model_output` dispatches:

```python
    def _model_output(self, encodings: dict[str, Any]):
        if self.tagger_backend == 'onnx':
            import numpy as np
            ort_inputs = {k: np.asarray(encodings[k], dtype=np.int64)
                          for k in ('input_ids', 'attention_mask',
                                    'token_type_ids')}
            (hidden,) = self.onnx_session.run(['last_hidden_state'],
                                              ort_inputs)
            sequence_output = torch.from_numpy(hidden).to(
                self.model.device)
            word_mask = torch.as_tensor(encodings['word_mask'],
                                        device=self.model.device)
            tag_logits, span_logits = self.model.classify(
                sequence_output, word_mask)
            return ChartClassifierOutput(tag_logits=tag_logits,
                                         span_logits=span_logits)
        ... existing torch path (autocast + self.model(**tensors)) ...
```

`_initialise_model` gains `tagger_backend='torch'` kwarg on
`BobcatParser.__init__` (docstring: 'onnx' runs the encoder via
onnxruntime; requires `tools/export_onnx.py` to have been run and the
`onnxruntime`/`onnxruntime-gpu` package). Session construction:

```python
        onnx_session = None
        if tagger_backend == 'onnx':
            try:
                import onnxruntime
            except ImportError as e:
                raise ImportError(
                    "tagger_backend='onnx' requires onnxruntime; pip "
                    'install onnxruntime (or onnxruntime-gpu)') from e
            onnx_path = self.model_dir / 'bobcat-body.onnx'
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f'{onnx_path} not found; run tools/export_onnx.py '
                    'first')
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                         if torch.device(self.device).type == 'cuda'
                         else ['CPUExecutionProvider'])
            onnx_session = onnxruntime.InferenceSession(
                str(onnx_path), providers=providers)
```

passed into `Tagger(..., tagger_backend=tagger_backend,
onnx_session=onnx_session)`.

- [ ] **Step 4: tests** (append to the parser test file):

```python
def test_invalid_tagger_backend():
    with pytest.raises(ValueError):
        BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                     tagger_backend='tensorflow')


def test_onnx_tagger_matches_torch(bobcat_parser):
    onnxruntime = pytest.importorskip('onnxruntime')
    import pathlib
    import subprocess
    import sys
    model_dir = pathlib.Path.home() / '.cache/lambeq/bobcat/bobcat'
    if not (model_dir / 'bobcat-body.onnx').exists():
        subprocess.run([sys.executable, 'tools/export_onnx.py'],
                       check=True)
    onnx_parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                               tagger_backend='onnx')
    sentences = ['Alice likes Bob', 'I do not like Bob']
    assert (onnx_parser.sentences2trees(
                sentences, verbose=VerbosityLevel.SUPPRESS.value)
            == bobcat_parser.sentences2trees(
                sentences, verbose=VerbosityLevel.SUPPRESS.value))
```

Install CPU onnxruntime first: `$PY -m pip install -q onnxruntime`.
The export takes a few minutes and ~1.4GB disk in the model cache —
acceptable, runs once. If the ONNX trees differ on these two trivial
sentences, that is a bug (fp32 CPU EP should match closely enough for
clear-margin parses), debug before weakening anything.

- [ ] **Step 5: run both-backend gauntlet + commit**

```bash
LAMBEQ_BOBCAT_BACKEND=rust $PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
git add tools/export_onnx.py lambeq/ tests/
git commit -m "Add opt-in ONNX Runtime lane for the Bobcat tagger encoder

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Benchmarks on three machines

**Files:**
- Modify: `benchmarks/bobcat_throughput.py`, `benchmarks/RESULTS.md`

- [ ] **Step 1: benchmark flags.** Add to `bobcat_throughput.py`:
`--compile-model` (store_true; sets `kwargs['compile_model'] = True`)
and `--tagger-backend` (default None; sets kwargs when given). NOTE the
GPU auto-defaults change what `config: {}` means on CUDA — print the
EFFECTIVE tagger settings: after constructing the parser add
`print(f'tagger: batch_size={parser.tagger.batch_size} dtype={parser.tagger.dtype} backend={parser.tagger.tagger_backend}')`
to the output block.

- [ ] **Step 2: local CPU sanity** (no perf claims):
`$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench_long.txt --num 40 --parser-backend rust` runs clean with the fused lane.

- [ ] **Step 3: goosander matrix** (check occupancy first per memory
etiquette; setup at `/cs/research/intelsys/discoviz/kinianlo/lambeq-bench`,
venv-lab, rsync branch + `uv pip install -p venv-lab/bin/python ./rust`,
`XDG_CACHE_HOME=$BASE/cache`; run export_onnx + onnxruntime-gpu install
`uv pip install -p ... onnxruntime-gpu`; if the CUDA EP fails to
initialise on this driver, record the failure verbatim and run the ONNX
lane on CPU EP only as a data point):

```
long corpus:  (defaults now auto: fp16 batch64)
  --device cuda --parser-backend rust
  --device cuda --parser-backend rust --compile-model
  --device cuda --parser-backend rust --tagger-backend onnx
  --device cuda --parser-backend python          (reference)
short corpus:
  --device cuda --parser-backend rust
  --device cuda --parser-backend rust --compile-model
```

Also re-run the component profile snippet (BERT body vs heads vs
post-processing) from the spec evidence section, now measuring
forward_topk vs parse_raw, to settle the pipelining question — append
conclusion to RESULTS.

- [ ] **Step 4: beaker job** (same pattern as previous jobs:
`bench_job` script on the share, lambeq-beaker clone `git pull`, rust
rebuild, `qsub -l h=!mitchell.local`; runs: long corpus python/rust/
rust+compile; onnx optional if onnxruntime installs cleanly on CentOS 7
— if not, note it).

- [ ] **Step 5: record + push.** Append a `## Tier 3 (tagger)` section
to RESULTS.md: matrix tables per machine, the corpus-gate status line
(937/937 on torch lanes; ONNX mismatch count with the <=1% bar), the
post-handoff component profile, and an honest verdict on compile/ONNX
(keep-or-document-only). Commit and `git push`.

```bash
git add benchmarks/
git commit -m "Record Tier 3 tagger benchmark results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

Report at the end: sent/s progression per machine (before Tier 3 ->
defaults -> +handoff -> best lane), whether targets (>=900 handoff,
stretch >=1200) were met, and the pipelining verdict.
