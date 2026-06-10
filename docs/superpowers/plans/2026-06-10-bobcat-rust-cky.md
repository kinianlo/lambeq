# bobcat_rs Rust CKY Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust (PyO3) chart-parser core producing trees identical to the Python implementation, >=10x faster single-thread, with rayon batch parallelism.

**Architecture:** maturin crate `rust/` exposing `bobcat_rs.RustChartParser`; Python keeps the existing `ChartParser` as oracle/fallback behind a `parser_backend` switch in `BobcatParser`. One Python↔Rust crossing per batch; Rust returns flat node lists; Python rebuilds the best `ParseTree` per sentence.

**Tech Stack:** Rust (edition 2021), PyO3 0.22+ (abi3-py310), rayon, maturin; pytest differential tests against the in-repo Python oracle.

**Spec:** `docs/superpowers/specs/2026-06-10-bobcat-rust-cky-design.md`

---

## Conventions (every task)

- Repo: `/home/kinianlo/projects/lambeq`, branch `bobcat-throughput`.
- `PY=~/.pyenv/versions/qnlp/bin/python`. Rust via `~/.cargo/bin` (Task 0).
- Bobcat model cached at `~/.cache/lambeq/bobcat/bobcat` (contains
  `grammar.json` and `config.json` with the `cats` list). Tests load the
  grammar from there via the `bobcat_parser` fixture or directly.
- Rebuild the extension after each Rust change:
  `$PY -m pip install -q ./rust` (or `maturin develop` inside `rust/`).
- All new pytest files start with `bobcat_rs = pytest.importorskip('bobcat_rs')`
  so the suite stays green when the extension isn't built.
- Commit style: plain sentence + trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **The Python files being ported are the single source of truth.** Where
  this plan says "port X exactly", the porting contract is the referenced
  Python code, and the differential test in the same task is the
  acceptance gate. Do not "improve" semantics while porting.

## File map

```
rust/
  pyproject.toml          maturin build config
  Cargo.toml
  src/lib.rs              PyO3 module: RustChartParser + debug_* functions
  src/category.rs         Atom, Feature, Category, markedup parser
  src/grammar.rs          rule tables built from grammar.json data
  src/tree.rs             Rule enum, Node (ParseTree equivalent), VarState, Unify
  src/rules.rs            combinators, punct, coordination, type raising/changing
  src/chart.rs            Cell, Chart (beam semantics)
  src/parser.rs           CKY loop, scoring, result_cats, parse_batch
lambeq/bobcat/rust_backend.py    RustBackend wrapper + nodes_to_tree
lambeq/text2diagram/model_based_reader/bobcat_parser.py   dispatch
tests/test_bobcat_rs.py          differential tests (importorskip)
benchmarks/check_equivalence.py  full-corpus equivalence script
benchmarks/bobcat_throughput.py  --parser-backend flag
```

---

### Task 0: Rust toolchain + crate scaffold

**Files:**
- Create: `rust/pyproject.toml`, `rust/Cargo.toml`, `rust/src/lib.rs`
- Modify: `.gitignore` (add `rust/target/`)

- [ ] **Step 1: Install toolchain**

```bash
which cargo || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
~/.cargo/bin/cargo --version
PY=~/.pyenv/versions/qnlp/bin/python
$PY -m pip install -q maturin
```

Expected: cargo >= 1.75 prints.

- [ ] **Step 2: Create the crate**

`rust/Cargo.toml`:

```toml
[package]
name = "bobcat_rs"
version = "0.1.0"
edition = "2021"

[lib]
name = "bobcat_rs"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["abi3-py310"] }
rayon = "1.10"

[profile.release]
lto = true
codegen-units = 1
```

`rust/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "bobcat_rs"
requires-python = ">=3.10"
version = "0.1.0"

[tool.maturin]
features = ["pyo3/extension-module"]
```

`rust/src/lib.rs`:

```rust
use pyo3::prelude::*;

#[pymodule]
fn bobcat_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

- [ ] **Step 3: Build and smoke-test**

```bash
echo 'rust/target/' >> .gitignore
cd /home/kinianlo/projects/lambeq && PATH=$PATH:~/.cargo/bin $PY -m pip install -q ./rust
$PY -c "import bobcat_rs; print(bobcat_rs.__version__)"
```

Expected: `0.1.0`.

- [ ] **Step 4: Commit**

```bash
git add rust/ .gitignore
git commit -m "Scaffold bobcat_rs PyO3 crate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 1: Categories (`category.rs`)

**Files:**
- Create: `rust/src/category.rs`; Modify: `rust/src/lib.rs`
- Test: `tests/test_bobcat_rs.py`

**Porting contract** — `lambeq/bobcat/lexicon.py`, replicated exactly:

- `Atom`: index order `['', 'N', 'NP', 'S', 'PP', 'conj', ',', ';', ':', '.', 'LQU', 'RQU', 'LRB', 'RRB']`; `is_punct = index >= 6`.
- `Feature`: index order from `lexicon.py:48-50`; `is_free = index <= 1` (NONE, X).
- `Category` struct: `atom: u8, feature: u8, var: u8, slot: Option<u8>` (the
  `Relation` is only ever needed for its slot number in `repr`; relation
  category strings are never compared), `dir: Dir (None|Fwd|Bwd)`,
  `result/argument: Option<Rc<Category>>` (or arena indices),
  `type_raising_dep_var: u8`, plus precomputed `hash: u64` and
  `vars: u16` bitset.
- `_hash` (lexicon.py:184-192): atomic -> hash of (atom, feature-with-X-
  mapped-to-NONE); complex -> hash of (result.hash, argument.hash, dir).
  Any deterministic 64-bit combiner is fine — both sides of every
  comparison are Rust, but X must hash like NONE so the S-feature
  equality classes share buckets.
- `_equals` (lexicon.py:197-211): hash short-circuit; atomic equality
  treats `S[free] == S[free]` where free = NONE or X; complex recurses.
- `_matches` (lexicon.py:217-225): NONE feature on self matches anything.
- `translate(var_map, feature)` (lexicon.py:124-148): X-feature
  substitution on atoms; vars relabelled through the map (missing var ->
  panic is acceptable: Python would KeyError; in practice var 0 must map
  to 0 — `Unify` always provides it; see Task 2).
- Markedup parser (lexicon.py:252-308): both regexes ported as a hand
  written scanner; `VARIABLES = '+_YZWVUTRQAB'` index = var id; the
  wrap-in-parens retry in `parse` (lexicon.py:267-277); slot counter
  increments only `if in_result` (lexicon.py:289-290);
  `type_raising_dep_var` applies only to the OUTERMOST slash
  (lexicon.py:298 — inner recursive calls pass 0).
- Plain string form `__str__` and full `__repr__` (lexicon.py:150-182)
  including the parenthesisation rule and `{var}` / `<slot>` suffixes —
  repr is the differential-test probe.

**PyO3 debug bindings** (add to `lib.rs`):

```rust
#[pyfunction]
fn debug_parse_category(s: &str, tr_var: &str) -> PyResult<(String, String)>
// returns (str(cat), repr(cat)) after parsing s with type_raising_dep_var tr_var
#[pyfunction]
fn debug_cat_eq(a: &str, b: &str) -> PyResult<bool>
#[pyfunction]
fn debug_cat_matches(a: &str, b: &str) -> PyResult<bool>
```

- [ ] **Step 1: Write the failing differential test**

Create `tests/test_bobcat_rs.py`:

```python
import itertools
import json
from pathlib import Path

import pytest

bobcat_rs = pytest.importorskip('bobcat_rs')

from lambeq.bobcat.lexicon import Category

MODEL_DIR = Path.home() / '.cache/lambeq/bobcat/bobcat'


@pytest.fixture(scope='module')
def grammar_data():
    with open(MODEL_DIR / 'grammar.json') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def all_category_strings(grammar_data):
    cats = set(grammar_data['categories'].keys())
    cats |= set(grammar_data['categories'].values())
    for left, right in grammar_data['binary_rules']:
        cats |= {left, right}
    for _, left, right, res, _ in grammar_data['type_changing_rules']:
        cats |= {left, res} | ({right} if right else set())
    for left, right, _ in grammar_data['type_raising_rules']:
        cats |= {left, right}
    return sorted(cats)


def test_category_parse_matches_python(all_category_strings):
    for s in all_category_strings:
        py = Category.parse(s)
        rs_str, rs_repr = bobcat_rs.debug_parse_category(s, '+')
        assert rs_str == str(py), s
        assert rs_repr == repr(py), s


def test_type_raising_var_parse(grammar_data):
    for _, tr_cat, var in grammar_data['type_raising_rules']:
        py = Category.parse(tr_cat, var)
        rs_str, rs_repr = bobcat_rs.debug_parse_category(tr_cat, var)
        assert (rs_str, rs_repr) == (str(py), repr(py)), (tr_cat, var)


def test_category_eq_and_matches(all_category_strings):
    sample = all_category_strings[::7][:60]
    for a, b in itertools.product(sample, repeat=2):
        pa, pb = Category.parse(a), Category.parse(b)
        assert bobcat_rs.debug_cat_eq(a, b) == (pa == pb), (a, b)
        assert bobcat_rs.debug_cat_matches(a, b) == pa.matches(pb), (a, b)
```

- [ ] **Step 2: Run to verify failure**

```bash
$PY -m pytest tests/test_bobcat_rs.py -x -q
```

Expected: AttributeError (no `debug_parse_category`) — i.e. fails, not skips.

- [ ] **Step 3: Implement `category.rs` per the contract; register the three debug functions in `lib.rs`; rebuild**

```bash
PATH=$PATH:~/.cargo/bin $PY -m pip install -q ./rust
```

- [ ] **Step 4: Run tests until green**

```bash
$PY -m pytest tests/test_bobcat_rs.py -q
```

Expected: 3 passed. Debug any mismatch by comparing `repr` outputs.

- [ ] **Step 5: Commit**

```bash
git add rust/ tests/test_bobcat_rs.py
git commit -m "Port Bobcat category semantics to Rust with differential tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Trees, unification, core combinators (`tree.rs`, `rules.rs` part 1)

**Files:**
- Create: `rust/src/tree.rs`, `rust/src/rules.rs`, `rust/src/grammar.rs`
- Modify: `rust/src/lib.rs`
- Test: `tests/test_bobcat_rs.py`

**Porting contract:**

`tree.rs` — from `lambeq/bobcat/tree.py`, with dependencies REMOVED but
variable state KEPT (it gates rules):

- `Rule` enum: exactly the 17 names of tree.py:207-225 (string names used
  in output node lists must match Python, e.g. `'BA'`, `'ADJ_CONJ'`).
- `VarState { filled: bool }`; `var_map: SmallVec/array indexed by var id
  (0..12), Option<VarState>` — presence in the map is significant, not
  just the flag (Python checks `var in var_map`).
- `Node { rule: Rule, cat: CatRef, left/right: Option<NodeRef>,
  var_map, score: f64, word: Option<(String, u32)> }` plus the derived
  flags ported from tree.py:254-271: `is_leaf`,
  `coordinated_or_type_raised` (CONJ|BTR|FTR), `coordinated` (CONJ),
  `bwd_comp` (BC|GBC), `fwd_comp` (FC|GFC).
- Constructors port the var_map logic ONLY (deps dropped):
  - `Lexical` (tree.py:295-300): var_map = {cat.var: filled}.
  - `Coordination` (tree.py:303-318): var_map = right's var_map with every
    entry `filled=false`.
  - `TypeChanging` (tree.py:321-347): head = left unless rule is LP (then
    right); var_map = {cat.var: head.variable} if cat.var != 0 AND head's
    var_map contains head.cat.var, else empty.
  - `PassThrough`/`LeftPunct`/`RightPunct`/`AdjectivalConj`
    (tree.py:350-372): var_map copied from the passthrough child.
  - `TypeRaising` (tree.py:375-388): var_map = {1: left.variable} if left
    has one, else empty; rule FTR if cat.fwd else BTR.
  - `BinaryCombinator` (tree.py:391-431): port the var_map merge loop
    (tree.py:396-406) exactly: for i in 1..num_variables, look up
    old_left[i] in left.var_map and old_right[i] in right.var_map; both
    present -> merged entry with `filled=true` (Python `__add__` result
    has default filled=True); one present -> `as_filled(True)`. Skip the
    dependency partitioning (tree.py:408-428) entirely.
- `Unify` (tree.py:113-204): port in full — trans_left/right,
  old_left/right, num_variables (starts 1), feature; `unify_recursive`
  including the S-feature X handling and the **double-filled rejection**
  (tree.py:170-177: if left.var and right.var both unseen and both
  var_maps contain them and both are filled -> fail); `add_vars` (iterate
  vars in the same order Python's set iteration... see NOTE);
  `get_new_outer_var`; `translate_arg/res`.

  **NOTE on `add_vars` ordering:** Python iterates `cat.vars` (a set of
  small ints). For var-id sets CPython iterates ints in increasing order
  for values < 2^60, so iterate ascending var id in Rust. The resulting
  numbering affects var ids only, not tree structure, but keep it
  deterministic and matching.

`grammar.rs`: plain-data model mirroring `Grammar` (grammar.py:24-67) and
the table construction from `Rules.__init__` (rules.py:87-124):
`rule_instances: HashSet<(CatRef, CatRef)>` (semantic hash/eq from Task 1),
`type_raising_rules: Vec<(Category, Vec<Category>)>` insertion-ordered,
`unary_rules`, `left/right_punct_type_changing_rules` — all
insertion-ordered maps (Python dicts) because `match_rule`
(rules.py:73-81) does exact lookup first, then a **linear scan in
insertion order** using `matches` — port that two-phase lookup exactly.

`rules.rs` part 1 — from `lambeq/bobcat/rules.py`:

- `CatKind::of` (rules.py:42-53) and `is_standard`.
- `combine` dispatch table (rules.py:126-169) verbatim, including the
  `None`-fallback chains and result-order (coordination before
  adjectival_conj, etc.).
- `backward_application`/`forward_application` (rules.py:248-266) with
  Eisner NF checks; `application` (rules.py:318-328).
- `composition` (rules.py:330-360) incl. the fallback to the generalised
  variants; `gc2`/`gc3` (rules.py:362-425);
  `generalised_forward_composition` (rules.py:427-445),
  `generalised_backward_composition` (rules.py:447-465 — including the
  `var_map[...].filled` gate and the `S[dcl]\S[dcl]` match),
  `generalised_backward_cross_composition` (rules.py:467-485). Python
  catches `AttributeError` from missing `.result`/`.argument` on atomic
  categories — in Rust, make those accessors return Option and bail out
  to `None` wherever Python would have raised.

**PyO3 debug binding** (in `lib.rs`, on the parser object built in Step 1
of the test below — constructor takes the grammar data only for now):

```rust
#[pyclass]
struct RustRules { ... }
#[pymethods] impl RustRules {
    #[new] fn new(categories: HashMap<String,String>,
                  binary_rules: Vec<(String,String)>,
                  type_changing_rules: Vec<(u32, String, Option<String>, String, bool)>,
                  type_raising_rules: Vec<(String, String, String)>,
                  eisner_normal_form: bool) -> PyResult<Self>;
    /// Combine two LEXICAL trees whose supertags are the given PLAIN
    /// category strings (looked up through the markedup `categories`
    /// table, like ChartParser.__call__ does); returns
    /// [(rule_name, plain_result_cat_str)].
    fn debug_combine(&self, left: &str, right: &str) -> PyResult<Vec<(String,String)>>;
}
```

- [ ] **Step 1: Write the failing differential test** (append to `tests/test_bobcat_rs.py`):

```python
@pytest.fixture(scope='module')
def py_rules(grammar_data):
    from lambeq.bobcat.grammar import Grammar
    from lambeq.bobcat.rules import Rules
    grammar = Grammar(**grammar_data)
    marked_up = {plain: Category.parse(marked)
                 for plain, marked in grammar.categories.items()}
    return Rules(True, grammar, marked_up), marked_up


@pytest.fixture(scope='module')
def rs_rules(grammar_data):
    return bobcat_rs.RustRules(grammar_data['categories'],
                               [tuple(r) for r in grammar_data['binary_rules']],
                               [tuple(r) for r in grammar_data['type_changing_rules']],
                               [tuple(r) for r in grammar_data['type_raising_rules']],
                               True)


def test_combine_matches_python_on_all_rule_instances(
        grammar_data, py_rules, rs_rules):
    from lambeq.bobcat.tree import Lexical
    rules, marked_up = py_rules
    checked = 0
    for left_str, right_str in grammar_data['binary_rules']:
        if left_str not in marked_up or right_str not in marked_up:
            continue
        left = Lexical(marked_up[left_str], 'l', 1)
        right = Lexical(marked_up[right_str], 'r', 2)
        expected = [(t.rule.name, str(t.cat))
                    for t in rules.combine(left, right)]
        got = rs_rules.debug_combine(left_str, right_str)
        assert got == expected, (left_str, right_str)
        checked += 1
    assert checked > 1000
```

- [ ] **Step 2: Run to verify failure** (`AttributeError: RustRules`), then implement `tree.rs`, `grammar.rs`, `rules.rs` part 1 per the contract; rebuild; iterate until green:

```bash
PATH=$PATH:~/.cargo/bin $PY -m pip install -q ./rust && $PY -m pytest tests/test_bobcat_rs.py -q
```

Expected: all passed, `checked > 1000`. Any mismatch: print the pair,
reproduce in a `python -i` session against rules.py, fix Rust.

- [ ] **Step 3: Commit**

```bash
git add rust/ tests/test_bobcat_rs.py
git commit -m "Port Bobcat unification and core combinators to Rust

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Remaining rules (punct, coordination, type raising/changing)

**Files:**
- Modify: `rust/src/rules.rs`, `rust/src/lib.rs`
- Test: `tests/test_bobcat_rs.py`

**Porting contract** — rest of rules.py:

- `type_raise` (rules.py:171-176) via `match_rule` over
  `type_raising_rules`.
- `type_change_cat`/`type_change` (rules.py:178-200).
- `left_punct` (rules.py:202-225): LeftPunct gate, the comma/semicolon
  coordination special case, left-punct type changing.
- `right_punct` (rules.py:227-246): RightPunct gate (`not
  left.coordinated_or_type_raised`), right-punct type changing gated on
  `not left.coordinated`.
- `coordination` (rules.py:301-308), `adjectival_conj` (rules.py:310-316).

Extend `RustRules` debug bindings:

```rust
fn debug_combine_full(&self, left: &str, right: &str) -> PyResult<Vec<(String,String)>>;
// same as debug_combine — punct/conj paths now reachable
fn debug_type_change(&self, cat: &str) -> PyResult<Vec<(String,String)>>;
fn debug_type_raise(&self, cat: &str) -> PyResult<Vec<(String,String)>>;
```

- [ ] **Step 1: Write the failing test** (append):

```python
def test_type_change_and_raise_match_python(grammar_data, py_rules, rs_rules):
    from lambeq.bobcat.tree import Lexical
    rules, marked_up = py_rules
    for cat_str, cat in list(marked_up.items()):
        tree = Lexical(cat, 'w', 1)
        expected_tc = [(t.rule.name, str(t.cat))
                       for t in rules.type_change([tree])]
        assert rs_rules.debug_type_change(cat_str) == expected_tc, cat_str
        expected_tr = [(t.rule.name, str(t.cat))
                       for t in rules.type_raise([tree])]
        assert rs_rules.debug_type_raise(cat_str) == expected_tr, cat_str


def test_punct_and_conj_combinations(grammar_data, py_rules, rs_rules):
    from lambeq.bobcat.tree import Lexical
    rules, marked_up = py_rules
    punct_or_conj = [s for s in marked_up
                     if marked_up[s].atom.is_punct
                     or str(marked_up[s].atom) == 'conj']
    others = [s for s in list(marked_up)[::5][:80]]
    pairs = ([(p, o) for p in punct_or_conj for o in others]
             + [(o, p) for p in punct_or_conj for o in others])
    for left_str, right_str in pairs:
        left = Lexical(marked_up[left_str], 'l', 1)
        right = Lexical(marked_up[right_str], 'r', 2)
        expected = [(t.rule.name, str(t.cat))
                    for t in rules.combine(left, right)]
        got = rs_rules.debug_combine_full(left_str, right_str)
        assert got == expected, (left_str, right_str)
```

NOTE: `rules.combine` early-exits on `rule_instances`, so most pairs
yield `[]` on both sides — that exit must be replicated, not bypassed.

- [ ] **Step 2: Implement, rebuild, iterate to green** (same build/test commands as Task 2).

- [ ] **Step 3: Commit**

```bash
git add rust/ tests/test_bobcat_rs.py
git commit -m "Port Bobcat punctuation, coordination and type-changing rules to Rust

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Chart, beam, serial CKY (`chart.rs`, `parser.rs`)

**Files:**
- Create: `rust/src/chart.rs`, `rust/src/parser.rs`; Modify: `rust/src/lib.rs`
- Test: `tests/test_bobcat_rs.py`

**Porting contract:**

`chart.rs` — from `lambeq/bobcat/parser.py:69-200`, the highest-fidelity-
risk code in the project. Port LITERALLY, preserving:

- `Cell.find` (parser.py:84-98): binary search over the descending list
  returning `mid` on exact score equality.
- `Cell.add` (parser.py:100-163): stable sort of `to_add` by descending
  score (Rust `sort_by` is stable); the early `break` when full and below
  the last score; the per-category replacement scan (forward from
  `find(old_score)` while scores equal, then backward); insertion at
  `find(score)`; the beam-trim block including `min_score` update and the
  `added -= ...` bookkeeping; return `added`.
- `Chart` (parser.py:166-200): `min_score` default -inf,
  `parse_tree_count` accumulation.

`parser.rs` — from `parser.py:313-520`:

- `result_cats` table construction (parser.py:345-362) incl. the
  `[conj]` rewrite; the table is keyed by (label, tuple-of-categories)
  using the SEMANTIC category hash/eq from Task 1 (this is how Python
  behaves: dict keyed on Category tuples).
- `__call__` (parser.py:391-467) as `parse_one`: the lexical-cell block
  (type_change/type_raise only when `len(sentence) > 1` and span scores
  exist; unary scoring for non-leaf results; root filter when single
  word); the span loop with `max_parse_trees` break placement
  (parser.py:423 breaks the `end` loop), the `missing-span` `continue`,
  `max_span_score` computation, the `min_score` early `break` in the
  right-trees loop, type change/raise gating `len(sentence) > span_length
  + 1`, root filtering on the last span, and unary-vs-binary scoring
  dispatch.
- `calc_score_unary` (parser.py:469-488), `calc_score_binary`
  (parser.py:490-509 — incl. the NP-without-feature retry),
  `get_span_score` (parser.py:511-520); `missing_cat_score`/
  `missing_span_score` = ln of config values, -inf on ValueError
  (i.e. value <= 0). All score arithmetic f64, additions in the same
  order as Python.
- `filter_root`/`set_root_cats` (parser.py:366-389).
- Output extraction: best tree = `chart[(0, n-1)].trees[0]`; serialize
  post-order (children before parent, left before right) into
  `Vec<(String rule_name, String plain_cat, Option<String> word, i64 left, i64 right)>`
  with -1 for absent children; root is the LAST node; leaves therefore
  appear in sentence order. No parse -> `None` for that sentence.

**PyO3 class** (replaces `RustRules` as the public API; keep `RustRules`
debug bindings working by delegating):

```rust
#[pyclass]
struct RustChartParser { ... }
#[pymethods] impl RustChartParser {
    #[new]
    #[pyo3(signature = (categories, binary_rules, type_changing_rules,
                        type_raising_rules, cats, root_cats,
                        eisner_normal_form, max_parse_trees, beam_size,
                        input_tag_score_weight, missing_cat_score,
                        missing_span_score))]
    fn new(...) -> PyResult<Self>;
    fn set_root_cats(&mut self, root_cats: Option<Vec<String>>) -> PyResult<()>;
    /// sentences: [(words, [[(plain_cat, logp)]], {(i, j): {cat_id: score}})]
    /// `num_threads` is accepted from the start (the Python wrapper
    /// passes it) but IGNORED until Task 6 adds rayon — serial for now.
    #[pyo3(signature = (sentences, num_threads = 0))]
    fn parse_batch(&self, py: Python<'_>, sentences: Vec<SentenceInput>,
                   num_threads: usize)
        -> PyResult<Vec<Option<Vec<NodeTuple>>>>;
}
```

- [ ] **Step 1: Write the failing equivalence test** (append):

```python
@pytest.fixture(scope='module')
def bobcat_parser():
    from lambeq import BobcatParser, VerbosityLevel
    return BobcatParser(verbose=VerbosityLevel.SUPPRESS.value)


@pytest.fixture(scope='module')
def rs_parser(grammar_data, bobcat_parser):
    return bobcat_rs.RustChartParser(
        grammar_data['categories'],
        [tuple(r) for r in grammar_data['binary_rules']],
        [tuple(r) for r in grammar_data['type_changing_rules']],
        [tuple(r) for r in grammar_data['type_raising_rules']],
        bobcat_parser.tagger.model.config.cats,
        None, True, 50000, 32, 1.0, 0.01, 1e-05)


SENTENCES = [
    'Alice likes Bob',
    'What Alice is and is not .',
    'I do not like Bob',
    'the old man sees a book about science in the park',
    'Alice likes Bob and the cat follows the dog while my neighbour '
    'reads a long and boring report before breakfast',
]


def _tagged(bobcat_parser, sentences):
    return bobcat_parser.tagger([s.split() for s in sentences],
                                verbose='suppress')


def _py_tree(bobcat_parser, sent, tags):
    si = bobcat_parser._prepare_sentence(sent, tags)
    return bobcat_parser.parser(si)[0]


def test_serial_parse_equivalence(bobcat_parser, rs_parser):
    from lambeq.text2diagram.model_based_reader.bobcat_parser import (
        BobcatParser)
    out = _tagged(bobcat_parser, SENTENCES)
    inputs = []
    for sent in out.sentences:
        si = bobcat_parser._prepare_sentence(sent, out.tags)
        supertags = [[(st.category, st.probability) for st in sts]
                     for sts in si.input_supertags]
        inputs.append((si.words, supertags, si.span_scores))
    results = rs_parser.parse_batch(inputs)
    for sent, nodes in zip(out.sentences, results):
        py_tree = _py_tree(bobcat_parser, sent, out.tags)
        py_ccg = BobcatParser._build_ccgtree(py_tree)
        from lambeq.bobcat.rust_backend import nodes_to_tree
        rs_ccg = BobcatParser._build_ccgtree(nodes_to_tree(nodes))
        assert rs_ccg == py_ccg, ' '.join(sent.words)
```

(`nodes_to_tree` is written in this task — see Step 3 — because the test
needs it; the rest of `rust_backend.py` comes in Task 5.)

- [ ] **Step 2: Implement `chart.rs`, `parser.rs`, bindings; rebuild.**

- [ ] **Step 3: Create `lambeq/bobcat/rust_backend.py` (first half)**

```python
# (standard lambeq copyright header, as in other files)
"""Bridge to the optional bobcat_rs Rust chart-parser extension."""

from __future__ import annotations

from lambeq.bobcat.lexicon import Category
from lambeq.bobcat.tree import IndexedWord, ParseTree, Rule, Variable

NodeTupleT = 'tuple[str, str, str | None, int, int]'


def nodes_to_tree(nodes: list[tuple[str, str, str | None, int, int]]
                  ) -> ParseTree:
    """Rebuild a ParseTree from a bobcat_rs flat node list.

    Nodes are in post-order (children first, leaves in sentence order,
    root last). The rebuilt tree carries no dependency state: it exists
    for `_build_ccgtree` (structure, rules, categories, leaf words) and
    for `metadata['original']`.
    """
    trees: list[ParseTree] = []
    index = 0
    for rule_name, cat_str, word, left, right in nodes:
        cat = Category.parse(cat_str)
        var_map = {}
        if word is not None:
            index += 1
            var_map = {cat.var: Variable(IndexedWord(word, index))}
        trees.append(ParseTree(Rule[rule_name],
                               cat,
                               trees[left] if left >= 0 else None,
                               trees[right] if right >= 0 else None,
                               [], [], var_map))
    return trees[-1]
```

- [ ] **Step 4: Iterate to green**

```bash
PATH=$PATH:~/.cargo/bin $PY -m pip install -q ./rust && $PY -m pytest tests/test_bobcat_rs.py -q
```

For a mismatching sentence, shrink it (parse prefixes) and diff the two
trees' node lists; the usual culprits are Cell tie order and
`max_parse_trees`/`min_score` break placement.

- [ ] **Step 5: Commit**

```bash
git add rust/ tests/test_bobcat_rs.py lambeq/bobcat/rust_backend.py
git commit -m "Add serial Rust CKY parser with chart-level equivalence test

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Python dispatch (`parser_backend`)

**Files:**
- Modify: `lambeq/bobcat/rust_backend.py` (second half)
- Modify: `lambeq/text2diagram/model_based_reader/bobcat_parser.py`
- Test: `tests/text2diagram/model_based_reader/test_bobcat_parser.py`

- [ ] **Step 1: Write the failing tests** (append to the existing parser test file):

```python
def test_invalid_parser_backend():
    with pytest.raises(ValueError):
        BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                     parser_backend='cobol')


def test_rust_backend_matches_python(bobcat_parser):
    pytest.importorskip('bobcat_rs')
    from lambeq.bobcat.rust_backend import RustBackend
    rust_parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                               parser_backend='rust')
    assert isinstance(rust_parser.parser, RustBackend)
    sentences = ['Alice likes Bob', 'What Alice is and is not .']
    assert (rust_parser.sentences2trees(
                sentences, verbose=VerbosityLevel.SUPPRESS.value)
            == bobcat_parser.sentences2trees(
                sentences, verbose=VerbosityLevel.SUPPRESS.value))
    # root-cat switching must work on the Rust backend too
    rust_parser.parser.set_root_cats(['NP'])
    try:
        tree = rust_parser.sentence2tree('I do')
        assert tree.biclosed_type == CCGType.NOUN_PHRASE
    finally:
        rust_parser.parser.set_root_cats(None)
```

(Note: constructing a second BobcatParser loads the BERT model again;
keep this single test doing all rust-dispatch assertions.)

- [ ] **Step 2: Implement.**

Append to `lambeq/bobcat/rust_backend.py`:

```python
class RustBackend:
    """Drop-in for ChartParser backed by bobcat_rs.

    Satisfies the pieces of the ChartParser interface that
    BobcatParser uses: __call__ (single sentence), parse_batch,
    set_root_cats.
    """

    def __init__(self,
                 grammar,                       # lambeq.bobcat.Grammar
                 cats: list[str],
                 root_cats,
                 eisner_normal_form: bool,
                 max_parse_trees: int,
                 beam_size: int,
                 input_tag_score_weight: float,
                 missing_cat_score: float,
                 missing_span_score: float) -> None:
        import bobcat_rs
        self._parser = bobcat_rs.RustChartParser(
            grammar.categories,
            [tuple(r) for r in grammar.binary_rules],
            [tuple(r) for r in grammar.type_changing_rules],
            [tuple(r) for r in grammar.type_raising_rules],
            list(cats),
            list(root_cats) if root_cats is not None else None,
            eisner_normal_form,
            max_parse_trees,
            beam_size,
            input_tag_score_weight,
            missing_cat_score,
            missing_span_score)

    def set_root_cats(self, root_cats) -> None:
        self._parser.set_root_cats(
            list(root_cats) if root_cats is not None else None)

    @staticmethod
    def _to_input(sentence):
        supertags = [[(st.category, st.probability) for st in sts]
                     for sts in sentence.input_supertags]
        return (sentence.words, supertags, sentence.span_scores)

    def parse_batch(self, sentences, num_threads: int = 0):
        """Parse Sentence objects; returns list of ParseTree | None."""
        results = self._parser.parse_batch(
            [self._to_input(s) for s in sentences], num_threads)
        return [nodes_to_tree(nodes) if nodes is not None else None
                for nodes in results]

    def __call__(self, sentence):
        return _SingleResult(self.parse_batch([sentence])[0])


class _SingleResult:
    """Minimal stand-in for ParseResult: indexable, falsy when empty."""

    def __init__(self, tree: ParseTree | None) -> None:
        self._tree = tree

    def __bool__(self) -> bool:
        return self._tree is not None

    def __getitem__(self, index: int) -> ParseTree:
        if self._tree is None or index != 0:
            raise IndexError(index)
        return self._tree
```

In `bobcat_parser.py`:

1. `__init__` gains `parser_backend: str = 'auto'` (documented:
   `'auto'`/`'rust'`/`'python'`; auto prefers Rust when importable),
   passed to `_initialise_model`.
2. In `_initialise_model`, after building `self.tagger`:

```python
        if parser_backend not in ('auto', 'rust', 'python'):
            raise ValueError(f'Invalid `parser_backend`: {parser_backend}')
        if parser_backend == 'auto':
            try:
                import bobcat_rs  # noqa: F401
                parser_backend = 'rust'
            except ImportError:
                parser_backend = 'python'
        self.parser_backend = parser_backend

        grammar = Grammar.load(self.model_dir / 'grammar.json')
        if parser_backend == 'rust':
            try:
                from lambeq.bobcat.rust_backend import RustBackend
                self.parser = RustBackend(grammar,
                                          self.tagger.model.config.cats,
                                          root_cats,
                                          **config['parser'])
            except ImportError as e:
                raise ImportError(
                    "parser_backend='rust' requires the bobcat_rs "
                    'extension; build it with `pip install ./rust` '
                    '(needs a Rust toolchain, see rustup.rs)') from e
        else:
            self.parser = ChartParser(grammar,
                                      self.tagger.model.config.cats,
                                      root_cats,
                                      **config['parser'])
```

   NOTE: `config['parser']` keys match the RustBackend signature
   (eisner_normal_form, max_parse_trees, beam_size,
   input_tag_score_weight, missing_cat_score, missing_span_score).
3. In `sentences2trees`, inside `if sentences_valid:` after tagging,
   add a batch path BEFORE the n_jobs dispatch:

```python
            if self.parser_backend == 'rust':
                if n_jobs != 1:
                    import warnings
                    warnings.warn('`n_jobs` is ignored with the Rust '
                                  'parser backend; it parallelises '
                                  'internally', stacklevel=2)
                sentence_inputs = [self._prepare_sentence(sent, tags)
                                   for sent in tag_results.sentences]
                try:
                    parse_trees = self.parser.parse_batch(sentence_inputs)
                except Exception as e:
                    # a Rust panic surfaces here; per-sentence failures
                    # come back as None, so this is a whole-batch bug
                    raise BobcatParseError(
                        ' '.join(tag_results.sentences[0].words)) from e
                for sent, tree in zip(tag_results.sentences, parse_trees):
                    if tree is not None:
                        trees.append(self._build_ccgtree(tree))
                    elif suppress_exceptions:
                        trees.append(None)
                    else:
                        raise BobcatParseError(' '.join(sent.words))
            elif n_jobs == 1:
                ... (existing serial loop, unchanged)
            else:
                ... (existing multiprocessing branch, unchanged)
```

4. Env-var override so the whole suite can be forced onto a backend
   without touching tests: in `__init__`, default
   `parser_backend = os.environ.get('LAMBEQ_BOBCAT_BACKEND', parser_backend)`
   only when the caller left it at `'auto'`.

- [ ] **Step 3: Run the new tests, then the full suite under BOTH backends**

```bash
$PY -m pytest tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
LAMBEQ_BOBCAT_BACKEND=python $PY -m pytest tests/test_bobcat.py tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
LAMBEQ_BOBCAT_BACKEND=rust   $PY -m pytest tests/test_bobcat.py tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

Expected: all green three times (64 tests in the parser+bobcat files).
Watch specifically: `test_root_filtering` and `test_failed_sentence`
mutate `bobcat_parser.parser` — they must pass with `RustBackend` too
(set_root_cats exists; `test_failed_sentence` replaces `self.parser`
with a raising callable, which the rust batch path never calls — confirm
the rust path goes through `self.parser.parse_batch`, so the monkeypatch
must still break it; if it doesn't, adjust the test expectation ONLY by
asking the controller first).

- [ ] **Step 4: Commit**

```bash
git add lambeq/bobcat/rust_backend.py \
    lambeq/text2diagram/model_based_reader/bobcat_parser.py \
    tests/text2diagram/model_based_reader/test_bobcat_parser.py
git commit -m "Dispatch BobcatParser between Rust and Python chart backends

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: rayon parallel `parse_batch`

**Files:**
- Modify: `rust/src/parser.rs`, `rust/src/lib.rs`
- Test: `tests/test_bobcat_rs.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_parse_batch_parallel_matches_serial(bobcat_parser, rs_parser):
    out = _tagged(bobcat_parser, SENTENCES * 8)
    inputs = []
    for sent in out.sentences:
        si = bobcat_parser._prepare_sentence(sent, out.tags)
        supertags = [[(st.category, st.probability) for st in sts]
                     for sts in si.input_supertags]
        inputs.append((si.words, supertags, si.span_scores))
    serial = rs_parser.parse_batch(inputs, 1)
    parallel = rs_parser.parse_batch(inputs, 0)   # 0 = all cores
    assert serial == parallel
```

- [ ] **Step 2: Implement.** `parse_batch(py, sentences, num_threads)`:
convert inputs to Rust structs while holding the GIL, then
`py.allow_threads(|| ...)` around the parse: `num_threads == 1` -> plain
iter; else a scoped rayon pool (`ThreadPoolBuilder::new()
.num_threads(if num_threads == 0 { 0 } else { num_threads })`) with
`par_iter` over sentences. The parser state must be `Sync` (immutable
after construction — interior caches, if any, must be per-sentence).
Convert node lists back to Python after re-acquiring the GIL.

- [ ] **Step 3: Rebuild, run the whole rs test file, then the suite under rust backend**

```bash
PATH=$PATH:~/.cargo/bin $PY -m pip install -q ./rust
$PY -m pytest tests/test_bobcat_rs.py -q
LAMBEQ_BOBCAT_BACKEND=rust $PY -m pytest tests/test_bobcat.py tests/text2diagram/model_based_reader/test_bobcat_parser.py -q
```

- [ ] **Step 4: Commit**

```bash
git add rust/ tests/test_bobcat_rs.py
git commit -m "Parallelise Rust parse_batch with rayon

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Full-corpus equivalence

**Files:**
- Create: `benchmarks/check_equivalence.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Verify Rust and Python Bobcat backends produce identical trees.

Usage: python benchmarks/check_equivalence.py sentences.txt [more.txt ...]
Exits non-zero and prints each mismatching sentence.
"""
import sys

from lambeq import BobcatParser, VerbosityLevel


def main() -> int:
    sentences = []
    for path in sys.argv[1:]:
        with open(path) as f:
            sentences += [line.split() for line in f if line.strip()]

    py = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                      parser_backend='python')
    rs = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                      parser_backend='rust')
    # share the tagger weights: drop the second BERT copy
    rs.tagger = py.tagger

    py_trees = py.sentences2trees(sentences, tokenised=True,
                                  suppress_exceptions=True)
    rs_trees = rs.sentences2trees(sentences, tokenised=True,
                                  suppress_exceptions=True)
    bad = 0
    for words, a, b in zip(sentences, py_trees, rs_trees):
        if a != b:
            bad += 1
            print('MISMATCH:', ' '.join(words))
    total = len(sentences)
    print(f'{total - bad}/{total} identical')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: Run on both corpora (regenerate them if /tmp was wiped — Task 1/Task 7 of the Tier-1 plan has the generators)**

```bash
$PY benchmarks/check_equivalence.py /tmp/bobcat_bench.txt /tmp/bobcat_bench_long.txt
```

Expected: `937/937 identical`, exit 0. Every mismatch is a bug: minimise
the sentence, diff the node lists, fix Rust (NOT the test), re-run.
This step is the identical-trees acceptance gate from the spec.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/check_equivalence.py
git commit -m "Add full-corpus Rust/Python equivalence checker

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Benchmarks and wrap-up

**Files:**
- Modify: `benchmarks/bobcat_throughput.py`, `benchmarks/RESULTS.md`

- [ ] **Step 1: Add `--parser-backend` to the benchmark**

In `benchmarks/bobcat_throughput.py`: add
`argp.add_argument('--parser-backend', default=None)`; include it in
`kwargs` when set (`kwargs['parser_backend'] = args.parser_backend`).
The stage-2 timing loop calls `parser.parser(sentence_input)` once per
sentence — keep it (it measures the serial path); ALSO time the batch
path when the backend is rust by adding after the existing loop:

```python
    if getattr(parser, 'parser_backend', 'python') == 'rust':
        start = time.perf_counter()
        parser.parser.parse_batch(
            [parser._prepare_sentence(s, tag_results.tags)
             for s in tag_results.sentences])
        batch_time = time.perf_counter() - start
        print(f'chart parse_batch: {batch_time:.2f}s '
              f'({rate(n, batch_time)})')
```

(place the print with the other stage prints; adjust ordering so output
stays grouped).

- [ ] **Step 2: Benchmark locally (CPU)**

```bash
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench_long.txt --batch-size 16 --parser-backend python
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench_long.txt --batch-size 16 --parser-backend rust
$PY benchmarks/bobcat_throughput.py /tmp/bobcat_bench.txt --batch-size 32 --parser-backend rust
```

Record in `benchmarks/RESULTS.md` under `## Rust CKY core (bobcat_rs)`,
noting serial chart time (python vs rust single-sentence loop) and the
`parse_batch` rayon time. The spec gate: rust serial >= 5x python serial
on the long corpus, else STOP and re-profile before proceeding.

- [ ] **Step 3: Benchmark on goosander (3090 Ti) and beaker (A40)**

Both machines have the repo + venvs on the project share
(`/cs/research/intelsys/discoviz/kinianlo/lambeq-bench` from the lab,
`/SAN/...` from beaker; see `benchmarks/RESULTS.md` headers). For each:
rsync the branch, install rustup + `pip install ./rust` into the
machine's venv (beaker: build ON the login node, CentOS 7 — rustup
supports glibc 2.17), then run the same three commands with
`--device cuda` prepended and record results. Check GPU occupancy first
on goosander (`nvidia-smi`); use `qsub` on beaker
(`bench_job.sh` pattern already in the share, excluding
`mitchell.local`).

- [ ] **Step 4: Commit, push, and report**

```bash
git add benchmarks/
git commit -m "Record Rust CKY benchmark results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

Report: speedups per machine, equivalence status (937/937), and whether
the chart parser is still the bottleneck anywhere.
