// Port of lambeq/bobcat/parser.py:313-520 — the serial CKY chart parser.
//
// Every break/continue and try/except in the Python `__call__` is mirrored
// here: the identical-trees guarantee depends on it.

use std::collections::HashMap;
use std::rc::Rc;

use crate::category::{self, Category, CatRef, ATOM_NP, FEATURE_NONE};
use crate::chart::Chart;
use crate::grammar::{CatKey, Grammar};
use crate::rules::Rules;
use crate::tree::{lexical, Node};

const NEG_INF: f64 = f64::NEG_INFINITY;

// result_cats labels (the Python strings 'conj' / 'unary' / 'binary').
const LABEL_CONJ: u8 = 0;
const LABEL_UNARY: u8 = 1;
const LABEL_BINARY: u8 = 2;

type ResultKey = (u8, Vec<CatKey>);

pub struct ChartParser {
    pub rules: Rules,
    max_parse_trees: isize,
    beam_size: usize,
    input_tag_score_weight: f64,
    missing_cat_score: f64,
    missing_span_score: f64,
    result_cats: HashMap<ResultKey, u32>,
    root_cats: Option<Vec<CatRef>>,
}

/// One serialised parse node: (rule_name, plain_cat, word, left_idx, right_idx).
pub type SerNode = (String, String, Option<String>, i64, i64);

impl ChartParser {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grammar: Grammar,
        cats: &[String],
        root_cats: Option<Vec<String>>,
        eisner_normal_form: bool,
        max_parse_trees: isize,
        beam_size: usize,
        input_tag_score_weight: f64,
        missing_cat_score: f64,
        missing_span_score: f64,
    ) -> ChartParser {
        let missing_cat_score = if missing_cat_score > 0.0 {
            missing_cat_score.ln()
        } else {
            NEG_INF
        };
        let missing_span_score = if missing_span_score > 0.0 {
            missing_span_score.ln()
        } else {
            NEG_INF
        };

        // result_cats (parser.py:345-362).
        const CONJ_TAG: &str = "[conj]";
        let mut result_cats: HashMap<ResultKey, u32> = HashMap::new();
        for (cat_id, cat_str) in cats.iter().enumerate() {
            let chain: Vec<&str> = cat_str.split("::").collect();
            if chain.len() == 1 && chain[0].ends_with(CONJ_TAG) {
                let base = &chain[0][..chain[0].len() - CONJ_TAG.len()];
                let base = if base.contains('/') || base.contains('\\') {
                    format!("({base})")
                } else {
                    base.to_string()
                };
                let cat_modified = format!("({base}\\{base})");
                let parsed = category::parse(&cat_modified, '+');
                result_cats.insert((LABEL_CONJ, vec![CatKey(parsed)]), cat_id as u32);
            } else {
                let parsed: Vec<CatKey> = chain
                    .iter()
                    .map(|s| CatKey(category::parse(s, '+')))
                    .collect();
                let label = if chain.len() > 1 { LABEL_UNARY } else { LABEL_BINARY };
                result_cats.insert((label, parsed), cat_id as u32);
            }
        }

        let rules = Rules::new(eisner_normal_form, grammar);
        let mut parser = ChartParser {
            rules,
            max_parse_trees,
            beam_size,
            input_tag_score_weight,
            missing_cat_score,
            missing_span_score,
            result_cats,
            root_cats: None,
        };
        parser.set_root_cats(root_cats);
        parser
    }

    /// parser.py:366-377 (plain parse path).
    pub fn set_root_cats(&mut self, root_cats: Option<Vec<String>>) {
        self.root_cats =
            root_cats.map(|cats| cats.iter().map(|s| category::parse(s, '+')).collect());
    }

    /// parser.py:379-389.
    fn filter_root(&self, trees: Vec<Rc<Node>>) -> Vec<Rc<Node>> {
        match &self.root_cats {
            None => trees,
            Some(roots) => trees
                .into_iter()
                .filter(|tree| roots.iter().any(|cat| cat.matches(&tree.cat)))
                .collect(),
        }
    }

    fn type_change_all(&self, trees: &[Rc<Node>]) -> Vec<Rc<Node>> {
        let mut out = Vec::new();
        for t in trees {
            out.extend(self.rules.type_change_node(t));
        }
        out
    }

    fn type_raise_all(&self, trees: &[Rc<Node>]) -> Vec<Rc<Node>> {
        let mut out = Vec::new();
        for t in trees {
            out.extend(self.rules.type_raise_node(t));
        }
        out
    }

    /// parser.py:391-467. Returns the best root tree, or None on failure.
    pub fn parse_one(
        &self,
        words: &[String],
        supertags: &[Vec<(String, f64)>],
        span_scores: &HashMap<(u32, u32), HashMap<u32, f64>>,
    ) -> Option<Rc<Node>> {
        let n = words.len();
        if n == 0 {
            return None;
        }
        let mut chart = Chart::new(self.beam_size);

        // Lexical cells.
        for i in 0..n {
            let mut results: Vec<Rc<Node>> = Vec::new();
            for (cat_str, prob) in &supertags[i] {
                let cat = match self.rules.grammar.categories.get(cat_str) {
                    Some(c) => c.clone(),
                    // Python KeyError -> the parse cannot proceed.
                    None => return None,
                };
                let node = lexical(cat, words[i].clone(), (i + 1) as u32);
                node.score.set(self.input_tag_score_weight * prob);
                results.push(node);
            }

            if let Some(ss) = span_scores.get(&(i as u32, i as u32)) {
                if n > 1 {
                    let tc = self.type_change_all(&results);
                    results.extend(tc);
                    let tr = self.type_raise_all(&results);
                    results.extend(tr);

                    for tree in &results {
                        if tree.left.is_some() {
                            self.calc_score_unary(tree, ss);
                        }
                    }
                }
            }

            if n == 1 {
                results = self.filter_root(results);
            }
            chart.add(i as u32, i as u32, results);
        }

        // Main CKY loop.
        for span_length in 1..n {
            for end in span_length..n {
                if chart.parse_tree_count > self.max_parse_trees {
                    break;
                }

                let start = end - span_length;

                let ss = match span_scores.get(&(start as u32, end as u32)) {
                    Some(s) => s,
                    None => continue,
                };

                let mut max_span_score = self.missing_cat_score.max(self.missing_span_score);
                for &v in ss.values() {
                    if v > max_span_score {
                        max_span_score = v;
                    }
                }

                for split in (start + 1)..=end {
                    let left_trees = match chart.trees(start as u32, (split - 1) as u32) {
                        Some(t) => t.clone(),
                        None => continue,
                    };
                    let right_trees = match chart.trees(split as u32, end as u32) {
                        Some(t) => t.clone(),
                        None => continue,
                    };

                    for left in &left_trees {
                        for right in &right_trees {
                            let max_score =
                                left.score.get() + right.score.get() + max_span_score;
                            if max_score < chart.min_score(start as u32, end as u32) {
                                break;
                            }

                            let mut results = self.rules.combine(left, right);

                            if !results.is_empty() && n > span_length + 1 {
                                let tc = self.type_change_all(&results);
                                results.extend(tc);
                                let tr = self.type_raise_all(&results);
                                results.extend(tr);
                            }

                            if span_length == n - 1 {
                                results = self.filter_root(results);
                            }

                            for tree in &results {
                                if tree.right.is_some() {
                                    self.calc_score_binary(tree, ss);
                                } else {
                                    self.calc_score_unary(tree, ss);
                                }
                            }
                            chart.add(start as u32, end as u32, results);
                        }
                    }
                }
            }
        }

        chart
            .trees(0, (n - 1) as u32)
            .and_then(|t| t.first().cloned())
    }

    /// parser.py:469-488.
    fn calc_score_unary(&self, tree: &Rc<Node>, span_scores: &HashMap<u32, f64>) {
        let left = tree.left.as_ref().unwrap();
        let (base, key): (Rc<Node>, ResultKey) = match (&left.right, &left.left) {
            (None, Some(ll)) => {
                let ll = ll.clone();
                let key = (
                    LABEL_UNARY,
                    vec![
                        CatKey(tree.cat.clone()),
                        CatKey(left.cat.clone()),
                        CatKey(ll.cat.clone()),
                    ],
                );
                (ll, key)
            }
            _ => {
                let key = (
                    LABEL_UNARY,
                    vec![CatKey(tree.cat.clone()), CatKey(left.cat.clone())],
                );
                (left.clone(), key)
            }
        };

        let base_score = match (&base.left, &base.right) {
            (Some(bl), Some(br)) => bl.score.get() + br.score.get(),
            _ => base.score.get(),
        };

        let cat_id = self.result_cats.get(&key).copied();
        tree.score
            .set(base_score + self.get_span_score(span_scores, cat_id));
    }

    /// parser.py:490-509.
    fn calc_score_binary(&self, tree: &Rc<Node>, span_scores: &HashMap<u32, f64>) {
        let cat_id: Option<u32> = if tree.coordinated() {
            self.result_cats
                .get(&(LABEL_CONJ, vec![CatKey(tree.cat.clone())]))
                .copied()
        } else {
            match self
                .result_cats
                .get(&(LABEL_BINARY, vec![CatKey(tree.cat.clone())]))
                .copied()
            {
                Some(id) => Some(id),
                None => {
                    if tree.cat.atom == ATOM_NP {
                        let np = Category::new_atomic(ATOM_NP, FEATURE_NONE, 0, false);
                        self.result_cats
                            .get(&(LABEL_BINARY, vec![CatKey(np)]))
                            .copied()
                    } else {
                        None
                    }
                }
            }
        };

        let score = tree.left.as_ref().unwrap().score.get()
            + tree.right.as_ref().unwrap().score.get()
            + self.get_span_score(span_scores, cat_id);
        tree.score.set(score);
    }

    /// parser.py:511-520.
    fn get_span_score(&self, span_scores: &HashMap<u32, f64>, cat_id: Option<u32>) -> f64 {
        match cat_id {
            None => self.missing_cat_score,
            Some(id) => match span_scores.get(&id) {
                Some(&s) => s,
                None => self.missing_span_score,
            },
        }
    }
}

/// Serialise a parse tree POST-ORDER (left subtree, right subtree, parent),
/// leaves in sentence order, root last. Children indices are -1 when absent.
pub fn serialize_tree(root: &Rc<Node>) -> Vec<SerNode> {
    let mut out = Vec::new();
    serialize_node(root, &mut out);
    out
}

fn serialize_node(node: &Rc<Node>, out: &mut Vec<SerNode>) -> i64 {
    let left_idx = match &node.left {
        Some(l) => serialize_node(l, out),
        None => -1,
    };
    let right_idx = match &node.right {
        Some(r) => serialize_node(r, out),
        None => -1,
    };
    let word = node.word.as_ref().map(|(w, _)| w.clone());
    out.push((
        node.rule.name().to_string(),
        node.cat.to_plain_str(),
        word,
        left_idx,
        right_idx,
    ));
    (out.len() - 1) as i64
}
