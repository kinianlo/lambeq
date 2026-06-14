// The two fallible #[pyfunction]s return Result<T, PyErr>.  pyo3's wrapper
// macro emits a `From<!> for PyErr` coercion that clippy flags as useless;
// suppress it here since there is no structural workaround.
#![allow(clippy::useless_conversion)]

use std::collections::HashMap;

use pyo3::prelude::*;
use rayon::prelude::*;

mod build;
mod category;
mod chart;
mod fdiagram;
mod grammar;
mod parser;
mod rules;
mod tree;

use pyo3::exceptions::PyValueError;
use numpy::PyUntypedArrayMethods;

use crate::build::{assemble, Node};
use crate::fdiagram::{atom, atom_name, atom_z, FDiagram, FTy};
use crate::grammar::Grammar;
use crate::parser::{chart_size, serialize_tree, ChartParser, SerNode};
use crate::rules::Rules;
use crate::tree::lexical;

type AtomList = Vec<(String, i32)>;
type PyNode = (u8, u8, u8, Vec<AtomList>, String);

fn decode_types(types: Vec<AtomList>) -> Vec<FTy> {
    types
        .into_iter()
        .map(|al| FTy(al.into_iter().map(|(nm, z)| atom(&nm, z)).collect()))
        .collect()
}

fn decode_node(p: PyNode) -> Node {
    let (node_type, rule_tag, arity, types, name) = p;
    Node { node_type, rule_tag, arity, types: decode_types(types), name }
}

type BoxExport = (String, AtomList, AtomList, u8, i32, bool, u32);
type DiagramExport = (AtomList, Vec<BoxExport>, AtomList);

fn ty_export(ty: &FTy) -> AtomList {
    ty.0.iter().map(|&a| (atom_name(a), atom_z(a))).collect()
}

/// A built fast-diagram, exported back to Python as flat tuples that
/// `convert.rs_to_fast` re-interns by (name, z).
#[pyclass]
struct RsDiagram {
    inner: FDiagram,
}

#[pymethods]
impl RsDiagram {
    fn export(&self) -> DiagramExport {
        let terms = self
            .inner
            .terms
            .iter()
            .map(|(b, off)| {
                (b.name.clone(), ty_export(&b.dom), ty_export(&b.cod),
                 b.kind, b.z, b.is_dagger, *off)
            })
            .collect();
        (ty_export(&self.inner.dom), terms, ty_export(&self.inner.cod))
    }
}

/// Build a batch of fast-diagrams from post-order build programs, in
/// parallel across rayon workers. Decoding to owned Rust runs under the
/// GIL; the parallel `assemble` touches no Python state.
#[pyfunction]
fn build_diagrams(programs: Vec<Vec<PyNode>>) -> Vec<RsDiagram> {
    let decoded: Vec<Vec<Node>> = programs
        .into_iter()
        .map(|prog| prog.into_iter().map(decode_node).collect())
        .collect();
    decoded
        .into_par_iter()
        .map(|nodes| RsDiagram { inner: assemble(&nodes) })
        .collect()
}

/// One sentence's parse input: (words, per-word supertags, span scores).
type SentenceInput = (
    Vec<String>,
    Vec<Vec<(String, f64)>>,
    HashMap<(u32, u32), HashMap<u32, f64>>,
);

/// Parse category string `s` with type_raising_dep_var = VARIABLES.index(tr_var).
/// Returns (plain_str, full_repr).
#[pyfunction]
fn debug_parse_category(s: &str, tr_var: &str) -> Result<(String, String), pyo3::PyErr> {
    let tr_char = match tr_var.chars().next() {
        Some(c) => c,
        None => return Err(pyo3::exceptions::PyValueError::new_err("tr_var must be non-empty")),
    };
    let cat = category::parse(s, tr_char);
    Ok((cat.to_plain_str(), cat.to_repr_str()))
}

/// Parse both strings (tr_var='+'), compare with equals.
#[pyfunction]
fn debug_cat_eq(a: &str, b: &str) -> bool {
    let ca = category::parse(a, '+');
    let cb = category::parse(b, '+');
    ca.equals(&cb)
}

/// Parse category (tr_var='+'), return sorted list of variable ids in the vars bitset.
#[pyfunction]
fn debug_cat_vars(s: &str) -> Vec<u8> {
    let cat = category::parse(s, '+');
    let vars = cat.vars;
    let mut result = Vec::new();
    for i in 0u8..32 {
        if vars & (1u32 << i) != 0 {
            result.push(i);
        }
    }
    result
}

/// Parse both strings (tr_var='+'), compare with matches (a.matches(b)).
#[pyfunction]
fn debug_cat_matches(a: &str, b: &str) -> bool {
    let ca = category::parse(a, '+');
    let cb = category::parse(b, '+');
    ca.matches(&cb)
}

/// The CCG rules, holding the parsed grammar tables.
///
/// `unsendable` because categories use `Rc` and a thread-local parse cache;
/// the object is only ever touched under the Python GIL.
#[pyclass(unsendable)]
struct RustRules {
    rules: Rules,
}

#[pymethods]
impl RustRules {
    #[new]
    fn new(
        categories: HashMap<String, String>,
        binary_rules: Vec<(String, String)>,
        type_changing_rules: Vec<(u32, String, Option<String>, String, bool)>,
        type_raising_rules: Vec<(String, String, String)>,
        eisner_normal_form: bool,
    ) -> PyResult<Self> {
        let grammar = Grammar::new(categories, &binary_rules, &type_changing_rules, &type_raising_rules);
        let rules = Rules::new(eisner_normal_form, grammar);
        Ok(RustRules { rules })
    }

    /// Build Lexical trees from the marked-up category table, combine them,
    /// and return `(rule_name, plain_result_category)` pairs.
    fn debug_combine(&self, left: &str, right: &str) -> Result<Vec<(String, String)>, pyo3::PyErr> {
        let cat_l = match self.rules.grammar.categories.get(left) {
            Some(c) => c.clone(),
            None => return Err(pyo3::exceptions::PyKeyError::new_err(format!("unknown category: {left}"))),
        };
        let cat_r = match self.rules.grammar.categories.get(right) {
            Some(c) => c.clone(),
            None => return Err(pyo3::exceptions::PyKeyError::new_err(format!("unknown category: {right}"))),
        };

        let lt = lexical(cat_l, "l".to_string(), 1);
        let rt = lexical(cat_r, "r".to_string(), 2);

        let trees = self.rules.combine(&lt, &rt);
        Ok(trees
            .into_iter()
            .map(|t| (t.rule.name().to_string(), t.cat.to_plain_str()))
            .collect())
    }

    /// Like debug_combine but returns (rule_name, FULL REPR of result cat).
    fn debug_combine_repr(&self, left: &str, right: &str) -> Result<Vec<(String, String)>, pyo3::PyErr> {
        let cat_l = match self.rules.grammar.categories.get(left) {
            Some(c) => c.clone(),
            None => return Err(pyo3::exceptions::PyKeyError::new_err(format!("unknown category: {left}"))),
        };
        let cat_r = match self.rules.grammar.categories.get(right) {
            Some(c) => c.clone(),
            None => return Err(pyo3::exceptions::PyKeyError::new_err(format!("unknown category: {right}"))),
        };

        let lt = lexical(cat_l, "l".to_string(), 1);
        let rt = lexical(cat_r, "r".to_string(), 2);

        let trees = self.rules.combine(&lt, &rt);
        Ok(trees
            .into_iter()
            .map(|t| (t.rule.name().to_string(), t.cat.to_repr_str()))
            .collect())
    }

    /// Apply unary type-changing rules (Rule::U) to a category.
    /// Returns (rule_name, repr) pairs.
    fn debug_type_change(&self, cat_str: &str) -> Result<Vec<(String, String)>, pyo3::PyErr> {
        let cat = match self.rules.grammar.categories.get(cat_str) {
            Some(c) => c.clone(),
            None => return Err(pyo3::exceptions::PyKeyError::new_err(format!("unknown category: {cat_str}"))),
        };
        let node = lexical(cat, "w".to_string(), 1);
        let results = self.rules.type_change_node(&node);
        Ok(results
            .into_iter()
            .map(|n| (n.rule.name().to_string(), n.cat.to_repr_str()))
            .collect())
    }

    /// Apply type-raising rules to a category.
    /// Returns (rule_name, repr) pairs.
    fn debug_type_raise(&self, cat_str: &str) -> Result<Vec<(String, String)>, pyo3::PyErr> {
        let cat = match self.rules.grammar.categories.get(cat_str) {
            Some(c) => c.clone(),
            None => return Err(pyo3::exceptions::PyKeyError::new_err(format!("unknown category: {cat_str}"))),
        };
        let node = lexical(cat, "w".to_string(), 1);
        let results = self.rules.type_raise_node(&node);
        Ok(results
            .into_iter()
            .map(|n| (n.rule.name().to_string(), n.cat.to_repr_str()))
            .collect())
    }
}

/// The Rust CKY chart parser.
///
/// The shared parser state (`Grammar`/`Rules`/`ChartParser`) holds only
/// `Arc<Category>`, hash maps, vectors and primitives, so it is `Send + Sync`
/// and can be read concurrently by rayon workers. The per-sentence parse
/// trees use `Rc`/`Cell`, but each `parse_one` builds and consumes its chart
/// entirely within a single rayon task — no `Rc<Node>` ever crosses a thread
/// boundary (only owned `SerNode` primitives escape).
#[pyclass]
struct RustChartParser {
    parser: ChartParser,
}

// Compile-time guarantee that the shared parser state is `Send + Sync`; this
// is what makes the `par_iter` / `allow_threads` boundary in `parse_batch`
// sound. If a future change reintroduces an `Rc`/`Cell` field into the shared
// tables, this will fail to compile.
#[cfg(test)]
mod sync_checks {
    use super::*;
    fn assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn chart_parser_is_send_sync() {
        assert_send_sync::<ChartParser>();
        assert_send_sync::<RustChartParser>();
    }
}

#[pymethods]
impl RustChartParser {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (categories, binary_rules, type_changing_rules,
                        type_raising_rules, cats, root_cats, eisner_normal_form,
                        max_parse_trees, beam_size, input_tag_score_weight,
                        missing_cat_score, missing_span_score))]
    fn new(
        categories: HashMap<String, String>,
        binary_rules: Vec<(String, String)>,
        type_changing_rules: Vec<(u32, String, Option<String>, String, bool)>,
        type_raising_rules: Vec<(String, String, String)>,
        cats: Vec<String>,
        root_cats: Option<Vec<String>>,
        eisner_normal_form: bool,
        max_parse_trees: i64,
        beam_size: usize,
        input_tag_score_weight: f64,
        missing_cat_score: f64,
        missing_span_score: f64,
    ) -> PyResult<Self> {
        let grammar = Grammar::new(
            categories,
            &binary_rules,
            &type_changing_rules,
            &type_raising_rules,
        );
        let parser = ChartParser::new(
            grammar,
            &cats,
            root_cats,
            eisner_normal_form,
            max_parse_trees as isize,
            beam_size,
            input_tag_score_weight,
            missing_cat_score,
            missing_span_score,
        );
        Ok(RustChartParser { parser })
    }

    #[pyo3(signature = (root_cats=None))]
    fn set_root_cats(&mut self, root_cats: Option<Vec<String>>) -> PyResult<()> {
        self.parser.set_root_cats(root_cats);
        Ok(())
    }

    /// Configure the raw top-k handoff lane. `tags` maps model tag ids to
    /// plain category strings (resolved to marked-up categories internally);
    /// the strategy strings are `'relative'` or `'absolute'`.
    fn configure_raw(
        &mut self,
        tags: Vec<String>,
        tag_prob_threshold: f64,
        tag_prob_threshold_strategy: String,
        span_prob_threshold: f64,
        span_prob_threshold_strategy: String,
    ) -> PyResult<()> {
        self.parser.configure_raw(
            &tags,
            tag_prob_threshold,
            tag_prob_threshold_strategy == "relative",
            span_prob_threshold,
            span_prob_threshold_strategy == "relative",
        );
        Ok(())
    }

    /// Parse a batch directly from the tagger's CPU top-k tensors, replicating
    /// `extract_topk` + `_prepare_sentence` in Rust (the fused lane).
    ///
    /// `tag_scores`/`tag_indices` have shape `[B, W, k_tag]` and
    /// `span_scores`/`span_indices` shape `[B, S, k_span]`, where `W` is the
    /// padded word count and `S` the padded span count. Requires a prior
    /// `configure_raw`. Same `num_threads` semantics as `parse_batch`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (words, tag_scores, tag_indices, span_scores, span_indices, num_threads = 0))]
    fn parse_batch_raw(
        &self,
        py: Python<'_>,
        words: Vec<Vec<String>>,
        tag_scores: numpy::PyReadonlyArray3<'_, f32>,
        tag_indices: numpy::PyReadonlyArray3<'_, i64>,
        span_scores: numpy::PyReadonlyArray3<'_, f32>,
        span_indices: numpy::PyReadonlyArray3<'_, i64>,
        num_threads: usize,
    ) -> PyResult<Vec<Option<Vec<SerNode>>>> {
        if !self.parser.is_raw_configured() {
            return Err(PyValueError::new_err(
                "parse_batch_raw requires configure_raw to be called first",
            ));
        }

        let b = words.len();
        let max_words = words.iter().map(|w| w.len()).max().unwrap_or(0);
        let max_spans = chart_size(max_words);

        let ts = tag_scores.shape();
        let ti = tag_indices.shape();
        let ss = span_scores.shape();
        let si = span_indices.shape();
        if ts[0] != b || ti[0] != b || ss[0] != b || si[0] != b {
            return Err(PyValueError::new_err(
                "batch dimension of the top-k arrays must equal len(words)",
            ));
        }
        if ts[1] < max_words || ti[1] < max_words {
            return Err(PyValueError::new_err(
                "tag arrays' word dimension is smaller than the longest sentence",
            ));
        }
        if ss[1] < max_spans || si[1] < max_spans {
            return Err(PyValueError::new_err(
                "span arrays' span dimension is smaller than chart_size of the \
                 longest sentence",
            ));
        }
        if ts[2] != ti[2] || ss[2] != si[2] {
            return Err(PyValueError::new_err(
                "scores and indices must share the same top-k dimension",
            ));
        }

        // numpy views cannot cross `allow_threads`; copy into owned, Send
        // arrays first (a memcpy, negligible next to the parsing work).
        let tag_scores = tag_scores.as_array().to_owned();
        let tag_indices = tag_indices.as_array().to_owned();
        let span_scores = span_scores.as_array().to_owned();
        let span_indices = span_indices.as_array().to_owned();

        let parse_one = |(sent, w): (usize, &Vec<String>)| {
            self.parser
                .parse_one_raw(
                    w,
                    sent,
                    &tag_scores,
                    &tag_indices,
                    &span_scores,
                    &span_indices,
                )
                .map(|tree| serialize_tree(&tree))
        };

        py.allow_threads(|| {
            if num_threads == 1 {
                Ok(words.iter().enumerate().map(parse_one).collect())
            } else {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads) // 0 = rayon default (all cores)
                    .build()
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "failed to build rayon thread pool: {e}"
                        ))
                    })?;
                Ok(pool.install(|| words.par_iter().enumerate().map(parse_one).collect()))
            }
        })
    }

    /// Parse a batch of sentences, optionally across rayon worker threads.
    ///
    /// `sentences`: `[(words, [[(plain_cat, logp)]], {(i, j): {cat_id: score}})]`.
    /// Each result is `None` (parse failure) or the post-order node list.
    ///
    /// `num_threads`: `1` → strictly serial (no rayon pool); `0` → rayon
    /// default (all cores); `n > 1` → a pool of `n` workers. The GIL is
    /// released for the whole parsing phase; results are converted back to
    /// Python objects only after the GIL is re-acquired.
    #[pyo3(signature = (sentences, num_threads = 0))]
    fn parse_batch(
        &self,
        py: Python<'_>,
        sentences: Vec<SentenceInput>,
        num_threads: usize,
    ) -> PyResult<Vec<Option<Vec<SerNode>>>> {
        let parse_one = |(words, supertags, span_scores): &SentenceInput| {
            self.parser
                .parse_one(words, supertags, span_scores)
                .map(|tree| serialize_tree(&tree))
        };

        py.allow_threads(|| {
            if num_threads == 1 {
                Ok(sentences.iter().map(parse_one).collect())
            } else {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads) // 0 = rayon default (all cores)
                    .build()
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "failed to build rayon thread pool: {e}"
                        ))
                    })?;
                Ok(pool.install(|| sentences.par_iter().map(parse_one).collect()))
            }
        })
    }
}

#[pymodule]
fn bobcat_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(debug_parse_category, m)?)?;
    m.add_function(wrap_pyfunction!(debug_cat_eq, m)?)?;
    m.add_function(wrap_pyfunction!(debug_cat_matches, m)?)?;
    m.add_function(wrap_pyfunction!(debug_cat_vars, m)?)?;
    m.add_class::<RustRules>()?;
    m.add_class::<RustChartParser>()?;
    m.add_class::<RsDiagram>()?;
    m.add_function(wrap_pyfunction!(build_diagrams, m)?)?;
    Ok(())
}
