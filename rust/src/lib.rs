use std::collections::HashMap;

use pyo3::prelude::*;

mod category;
mod grammar;
mod rules;
mod tree;

use crate::grammar::Grammar;
use crate::rules::Rules;
use crate::tree::lexical;

/// Parse category string `s` with type_raising_dep_var = VARIABLES.index(tr_var).
/// Returns (plain_str, full_repr).
#[pyfunction]
fn debug_parse_category(s: &str, tr_var: &str) -> PyResult<(String, String)> {
    let tr_char = tr_var
        .chars()
        .next()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("tr_var must be non-empty"))?;
    let cat = category::parse(s, tr_char);
    Ok((cat.to_plain_str(), cat.to_repr_str()))
}

/// Parse both strings (tr_var='+'), compare with equals.
#[pyfunction]
fn debug_cat_eq(a: &str, b: &str) -> PyResult<bool> {
    let ca = category::parse(a, '+');
    let cb = category::parse(b, '+');
    Ok(ca.equals(&cb))
}

/// Parse category (tr_var='+'), return sorted list of variable ids in the vars bitset.
#[pyfunction]
fn debug_cat_vars(s: &str) -> PyResult<Vec<u8>> {
    let cat = category::parse(s, '+');
    let vars = cat.vars;
    let mut result = Vec::new();
    for i in 0u8..32 {
        if vars & (1u32 << i) != 0 {
            result.push(i);
        }
    }
    Ok(result)
}

/// Parse both strings (tr_var='+'), compare with matches (a.matches(b)).
#[pyfunction]
fn debug_cat_matches(a: &str, b: &str) -> PyResult<bool> {
    let ca = category::parse(a, '+');
    let cb = category::parse(b, '+');
    Ok(ca.matches(&cb))
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
        _type_changing_rules: Vec<(u32, String, Option<String>, String, bool)>,
        _type_raising_rules: Vec<(String, String, String)>,
        eisner_normal_form: bool,
    ) -> PyResult<Self> {
        // type_changing / type_raising tables are accepted but UNUSED (Task 3).
        let grammar = Grammar::new(categories, &binary_rules);
        let rules = Rules::new(eisner_normal_form, grammar);
        Ok(RustRules { rules })
    }

    /// Build Lexical trees from the marked-up category table, combine them,
    /// and return `(rule_name, plain_result_category)` pairs.
    fn debug_combine(&self, left: &str, right: &str) -> PyResult<Vec<(String, String)>> {
        let cat_l = self
            .rules
            .grammar
            .categories
            .get(left)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("unknown category: {left}"))
            })?
            .clone();
        let cat_r = self
            .rules
            .grammar
            .categories
            .get(right)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("unknown category: {right}"))
            })?
            .clone();

        let lt = lexical(cat_l, "l".to_string(), 1);
        let rt = lexical(cat_r, "r".to_string(), 2);

        let trees = self.rules.combine(&lt, &rt);
        Ok(trees
            .into_iter()
            .map(|t| (t.rule.name().to_string(), t.cat.to_plain_str()))
            .collect())
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
    Ok(())
}
