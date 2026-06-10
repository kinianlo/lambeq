// The two fallible #[pyfunction]s return Result<T, PyErr>.  pyo3's wrapper
// macro emits a `From<!> for PyErr` coercion that clippy flags as useless;
// suppress it here since there is no structural workaround.
#![allow(clippy::useless_conversion)]

use std::collections::HashMap;

use pyo3::prelude::*;
use rayon::prelude::*;

mod category;
mod chart;
mod grammar;
mod parser;
mod rules;
mod tree;

use crate::grammar::Grammar;
use crate::parser::{serialize_tree, ChartParser, SerNode};
use crate::rules::Rules;
use crate::tree::lexical;

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
    Ok(())
}
