// Port of the grammar data shapes needed for rule application
// (grammar.py + the table construction in rules.py:87-124).

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::category::{self, CatRef};

/// A hash-set key wrapping a category, using SEMANTIC equality and hash
/// (delegating to `Category::equals` and the precomputed semantic hash).
#[derive(Clone)]
pub struct CatKey(pub CatRef);

impl PartialEq for CatKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.equals(&other.0)
    }
}

impl Eq for CatKey {}

impl Hash for CatKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.hash);
    }
}

// ---------------------------------------------------------------------------
// OrderedCatMap (rules.py:73-81 match_rule semantics)
// ---------------------------------------------------------------------------

/// An insertion-ordered map from Category keys (by semantic equality) to V.
///
/// Supports two lookup modes:
///  - `exact_get`: plain semantic-equality lookup (like Python dict.__getitem__)
///  - `match_rule_get`: exact first, then first insertion-order entry whose
///    key `.matches(cat)` (rules.py:73-81)
pub struct OrderedCatMap<V> {
    pub entries: Vec<(CatRef, V)>,
}

impl<V> Default for OrderedCatMap<V> {
    fn default() -> Self {
        OrderedCatMap { entries: Vec::new() }
    }
}

impl<V> OrderedCatMap<V> {
    pub fn new() -> Self {
        OrderedCatMap::default()
    }

    /// Find the entry with key semantically equal to `key`, or insert a new
    /// default entry. Returns a mutable reference to the value.
    pub fn entry_or_default(&mut self, key: CatRef) -> &mut V
    where
        V: Default,
    {
        // Linear scan for exact match (semantic equality).
        for i in 0..self.entries.len() {
            if self.entries[i].0.equals(&key) {
                return &mut self.entries[i].1;
            }
        }
        // Not found — append.
        self.entries.push((key, V::default()));
        let last = self.entries.len() - 1;
        &mut self.entries[last].1
    }

    /// Exact semantic-equality lookup (no matches fallback).
    /// Mirrors plain Python dict.__getitem__ semantics.
    pub fn exact_get(&self, cat: &CatRef) -> Option<&V> {
        for (k, v) in &self.entries {
            if k.equals(cat) {
                return Some(v);
            }
        }
        None
    }

    /// match_rule semantics (rules.py:73-81):
    ///  1. Exact semantic-equality lookup.
    ///  2. First insertion-order entry whose key `.matches(cat)`.
    ///  3. None if nothing matches.
    pub fn match_rule_get(&self, cat: &CatRef) -> Option<&V> {
        // Exact lookup first.
        for (k, v) in &self.entries {
            if k.equals(cat) {
                return Some(v);
            }
        }
        // Fallback: first entry whose key matches.
        for (k, v) in &self.entries {
            if k.matches(cat) {
                return Some(v);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// TypeChangingRule
// ---------------------------------------------------------------------------

/// A single entry in the type-changing rule tables.
pub struct TypeChangingRule {
    #[allow(dead_code)] // used in Task 4 chart scoring
    pub rule_id: u32,
    /// The result category (from `marked_up_categories[res_str]`).
    pub category: CatRef,
    #[allow(dead_code)] // used in Task 4 chart scoring
    pub replace: bool,
}

// ---------------------------------------------------------------------------
// Grammar
// ---------------------------------------------------------------------------

pub struct Grammar {
    /// plain category string -> parsed marked-up category.
    pub categories: HashMap<String, CatRef>,
    /// Set of binary rule instances (semantic keys), from the PLAIN strings.
    pub rule_instances: HashSet<(CatKey, CatKey)>,

    /// rules.py:96-100 — type-raising table.
    /// key = plain-parsed Category; values = list of tr_cat parsed with tr_var.
    pub type_raising_rules: OrderedCatMap<Vec<CatRef>>,

    /// rules.py:103-113 — unary type-changing rules (right is None).
    pub unary_rules: OrderedCatMap<Vec<TypeChangingRule>>,

    /// rules.py:114-117 — left-punctuation type-changing rules.
    /// Outer key = left Category (must be punct); inner key = right Category.
    /// Outer lookup: exact only (plain dict.__getitem__ in rules.py:216).
    /// Inner lookup: match_rule (type_change_cat calls match_rule).
    pub left_punct_type_changing_rules: OrderedCatMap<OrderedCatMap<Vec<TypeChangingRule>>>,

    /// rules.py:118-121 — right-punctuation type-changing rules.
    /// Outer key = right Category; inner key = left Category.
    pub right_punct_type_changing_rules: OrderedCatMap<OrderedCatMap<Vec<TypeChangingRule>>>,
}

impl Grammar {
    pub fn new(
        categories_raw: HashMap<String, String>,
        binary_rules: &[(String, String)],
        type_changing_rules: &[(u32, String, Option<String>, String, bool)],
        type_raising_rules: &[(String, String, String)],
    ) -> Grammar {
        // Build categories first — needed for type-changing result lookup.
        let categories: HashMap<String, CatRef> = categories_raw
            .into_iter()
            .map(|(plain, marked)| (plain, category::parse(&marked, '+')))
            .collect();

        let rule_instances = binary_rules
            .iter()
            .map(|(l, r)| {
                (
                    CatKey(category::parse(l, '+')),
                    CatKey(category::parse(r, '+')),
                )
            })
            .collect();

        // --- type_raising_rules (rules.py:96-100) ---
        let mut tr_rules: OrderedCatMap<Vec<CatRef>> = OrderedCatMap::new();
        for (cat_str, tr_cat_str, var_str) in type_raising_rules {
            let cat = category::parse(cat_str, '+');
            let tr_char = var_str.chars().next().unwrap_or('+');
            let tr_cat = category::parse(tr_cat_str, tr_char);
            tr_rules.entry_or_default(cat).push(tr_cat);
        }

        // --- type_changing_rules (rules.py:102-124) ---
        let mut unary_rules: OrderedCatMap<Vec<TypeChangingRule>> = OrderedCatMap::new();
        let mut left_punct_tcr: OrderedCatMap<OrderedCatMap<Vec<TypeChangingRule>>> =
            OrderedCatMap::new();
        let mut right_punct_tcr: OrderedCatMap<OrderedCatMap<Vec<TypeChangingRule>>> =
            OrderedCatMap::new();

        for (rule_id, left_str, right_str, res_str, replace) in type_changing_rules {
            let left = category::parse(left_str, '+');
            let right = right_str.as_ref().map(|s| category::parse(s, '+'));

            // res = marked_up_categories[res_str]
            let res = categories
                .get(res_str.as_str())
                .unwrap_or_else(|| panic!("result category not in table: {res_str}"))
                .clone();

            let tc_rule = TypeChangingRule {
                rule_id: *rule_id,
                category: res,
                replace: *replace,
            };

            match right {
                None => {
                    unary_rules.entry_or_default(left).push(tc_rule);
                }
                Some(right_cat) => {
                    if left.atom_is_punct() {
                        // left_punct_type_changing_rules[left][right].append(rule)
                        left_punct_tcr
                            .entry_or_default(left)
                            .entry_or_default(right_cat)
                            .push(tc_rule);
                    } else {
                        // right_punct_type_changing_rules[right][left].append(rule)
                        right_punct_tcr
                            .entry_or_default(right_cat)
                            .entry_or_default(left)
                            .push(tc_rule);
                    }
                }
            }
        }

        Grammar {
            categories,
            rule_instances,
            type_raising_rules: tr_rules,
            unary_rules,
            left_punct_type_changing_rules: left_punct_tcr,
            right_punct_type_changing_rules: right_punct_tcr,
        }
    }

    /// Whether `(left, right)` is a known binary rule instance.
    pub fn has_rule_instance(&self, left: &CatRef, right: &CatRef) -> bool {
        let key = (CatKey(left.clone()), CatKey(right.clone()));
        self.rule_instances.contains(&key)
    }
}
