use pyo3::prelude::*;

mod category;

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

/// Parse both strings (tr_var='+'), compare with matches (a.matches(b)).
#[pyfunction]
fn debug_cat_matches(a: &str, b: &str) -> PyResult<bool> {
    let ca = category::parse(a, '+');
    let cb = category::parse(b, '+');
    Ok(ca.matches(&cb))
}

#[pymodule]
fn bobcat_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(debug_parse_category, m)?)?;
    m.add_function(wrap_pyfunction!(debug_cat_eq, m)?)?;
    m.add_function(wrap_pyfunction!(debug_cat_matches, m)?)?;
    Ok(())
}
