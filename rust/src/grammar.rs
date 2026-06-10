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

pub struct Grammar {
    /// plain category string -> parsed marked-up category.
    pub categories: HashMap<String, CatRef>,
    /// Set of binary rule instances (semantic keys), from the PLAIN strings.
    pub rule_instances: HashSet<(CatKey, CatKey)>,
}

impl Grammar {
    pub fn new(
        categories: HashMap<String, String>,
        binary_rules: &[(String, String)],
    ) -> Grammar {
        let categories = categories
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

        Grammar {
            categories,
            rule_instances,
        }
    }

    /// Whether `(left, right)` is a known binary rule instance.
    pub fn has_rule_instance(&self, left: &CatRef, right: &CatRef) -> bool {
        let key = (CatKey(left.clone()), CatKey(right.clone()));
        self.rule_instances.contains(&key)
    }
}
