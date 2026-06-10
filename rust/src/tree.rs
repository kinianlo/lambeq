// Port of lambeq/bobcat/tree.py: parse-tree variable machinery,
// unification, and the ParseTree constructors.
//
// Dependency tracking has been REMOVED (it does not gate rule application);
// only the variable state is kept, because it gates which rules fire.

use std::cell::Cell;
use std::rc::Rc;

use crate::category::{CatRef, Category, FEATURE_NONE, FEATURE_X, ATOM_S};

/// Maximum variable id + 1. Marked-up categories use ids < 12, but
/// composition can introduce more; 32 is a safe upper bound.
pub const VAR_SLOTS: usize = 32;

/// A tree's variable map. Index = var id (< 32); `Some(filled)` means the
/// var is present with the given `filled` flag, `None` means absent.
///
/// The Python `Variable.fillers` lists are never read for gating, so the
/// only payload kept is the `filled` boolean.
pub type VarMap = [Option<bool>; VAR_SLOTS];

#[inline]
pub fn empty_var_map() -> VarMap {
    [None; VAR_SLOTS]
}

// ---------------------------------------------------------------------------
// Rule
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types, dead_code, clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rule {
    NONE,
    L,
    U,
    BA,
    FA,
    BC,
    FC,
    BX,
    GBC,
    GFC,
    GBX,
    LP,
    RP,
    BTR,
    FTR,
    CONJ,
    ADJ_CONJ,
}

impl Rule {
    /// The exact Python `Rule` member name (used for differential output).
    pub fn name(self) -> &'static str {
        match self {
            Rule::NONE => "NONE",
            Rule::L => "L",
            Rule::U => "U",
            Rule::BA => "BA",
            Rule::FA => "FA",
            Rule::BC => "BC",
            Rule::FC => "FC",
            Rule::BX => "BX",
            Rule::GBC => "GBC",
            Rule::GFC => "GFC",
            Rule::GBX => "GBX",
            Rule::LP => "LP",
            Rule::RP => "RP",
            Rule::BTR => "BTR",
            Rule::FTR => "FTR",
            Rule::CONJ => "CONJ",
            Rule::ADJ_CONJ => "ADJ_CONJ",
        }
    }
}

// ---------------------------------------------------------------------------
// Node (ParseTree)
// ---------------------------------------------------------------------------

#[allow(dead_code)] // left/right/score/word consumed by the chart in Task 4
pub struct Node {
    pub rule: Rule,
    pub cat: CatRef,
    pub left: Option<Rc<Node>>,
    pub right: Option<Rc<Node>>,
    pub var_map: VarMap,
    /// Mutated after construction (parser scoring), matching the Python
    /// oracle. Cell makes Node !Sync: nodes must stay within a single
    /// rayon task (parallelism is across sentences, never within one).
    pub score: std::cell::Cell<f64>,
    /// `(word, index)` for leaves; `None` otherwise.
    pub word: Option<(String, u32)>,
}

impl Node {
    #[allow(dead_code)]
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.rule == Rule::L
    }

    #[inline]
    pub fn coordinated_or_type_raised(&self) -> bool {
        matches!(self.rule, Rule::CONJ | Rule::BTR | Rule::FTR)
    }

    #[inline]
    pub fn coordinated(&self) -> bool {
        self.rule == Rule::CONJ
    }

    #[inline]
    pub fn bwd_comp(&self) -> bool {
        matches!(self.rule, Rule::BC | Rule::GBC)
    }

    #[inline]
    pub fn fwd_comp(&self) -> bool {
        matches!(self.rule, Rule::FC | Rule::GFC)
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// tree.py:295-300 (`Lexical`). var_map = {cat.var: filled=true}.
pub fn lexical(cat: CatRef, word: String, index: u32) -> Rc<Node> {
    debug_assert!(cat.var != 0, "lexical category must have a variable");
    let mut var_map = empty_var_map();
    var_map[cat.var as usize] = Some(true);
    Rc::new(Node {
        rule: Rule::L,
        cat,
        left: None,
        right: None,
        var_map,
        score: Cell::new(0.0),
        word: Some((word, index)),
    })
}

/// tree.py:303-318 (`Coordination`). var_map = right's var_map with every
/// present entry's `filled` set to false.
pub fn coordination(cat: CatRef, left: Rc<Node>, right: Rc<Node>) -> Rc<Node> {
    let mut var_map = empty_var_map();
    for (slot, right_slot) in var_map.iter_mut().zip(right.var_map.iter()) {
        if right_slot.is_some() {
            *slot = Some(false);
        }
    }
    Rc::new(Node {
        rule: Rule::CONJ,
        cat,
        left: Some(left),
        right: Some(right),
        var_map,
        score: Cell::new(0.0),
        word: None,
    })
}

/// tree.py:321-347 (`TypeChanging`), variable logic only.
#[allow(dead_code)] // TODO(Task 3): wired by type-changing/raising rules
pub fn type_changing(
    rule: Rule,
    cat: CatRef,
    left: Rc<Node>,
    right: Option<Rc<Node>>,
) -> Rc<Node> {
    let head: &Rc<Node> = if rule != Rule::LP {
        &left
    } else {
        right.as_ref().expect("LP type-changing needs a right child")
    };
    // head.variable = head.var_map[head.cat.var], absent -> None
    let outer_var = head.var_map[head.cat.var as usize];
    let mut var_map = empty_var_map();
    if cat.var != 0 {
        if let Some(filled) = outer_var {
            var_map[cat.var as usize] = Some(filled);
        }
    }
    Rc::new(Node {
        rule,
        cat,
        left: Some(left),
        right,
        var_map,
        score: Cell::new(0.0),
        word: None,
    })
}

/// tree.py:350-372 (`PassThrough`). cat and var_map taken from `passthrough`.
pub fn pass_through(
    rule: Rule,
    left: Rc<Node>,
    right: Rc<Node>,
    passthrough: &Rc<Node>,
) -> Rc<Node> {
    Rc::new(Node {
        rule,
        cat: passthrough.cat.clone(),
        var_map: passthrough.var_map,
        left: Some(left),
        right: Some(right),
        score: Cell::new(0.0),
        word: None,
    })
}

/// LeftPunct (tree.py:363-364): LP, pass through the right child.
pub fn left_punct_tree(left: Rc<Node>, right: Rc<Node>) -> Rc<Node> {
    let pt = right.clone();
    pass_through(Rule::LP, left, right, &pt)
}

/// RightPunct (tree.py:367-368): RP, pass through the left child.
pub fn right_punct_tree(left: Rc<Node>, right: Rc<Node>) -> Rc<Node> {
    let pt = left.clone();
    pass_through(Rule::RP, left, right, &pt)
}

/// AdjectivalConj (tree.py:371-372): ADJ_CONJ, pass through the right child.
pub fn adjectival_conj_tree(left: Rc<Node>, right: Rc<Node>) -> Rc<Node> {
    let pt = right.clone();
    pass_through(Rule::ADJ_CONJ, left, right, &pt)
}

/// tree.py:375-388 (`TypeRaising`). var_map = {1: left.var_map[left.cat.var]}
/// when present. Rule FTR if cat.fwd else BTR.
#[allow(dead_code)] // TODO(Task 3): wired by type-changing/raising rules
pub fn type_raising(cat: CatRef, left: Rc<Node>) -> Rc<Node> {
    let mut var_map = empty_var_map();
    if let Some(filled) = left.var_map[left.cat.var as usize] {
        var_map[1] = Some(filled);
    }
    let rule = if cat.fwd() { Rule::FTR } else { Rule::BTR };
    Rc::new(Node {
        rule,
        cat,
        left: Some(left),
        right: None,
        var_map,
        score: Cell::new(0.0),
        word: None,
    })
}

/// tree.py:391-431 (`BinaryCombinator`), variable-map merge only
/// (tree.py:396-406). The dependency partitioning is skipped.
pub fn binary_combinator(
    rule: Rule,
    cat: CatRef,
    left: Rc<Node>,
    right: Rc<Node>,
    u: &Unify,
) -> Rc<Node> {
    let mut var_map = empty_var_map();
    for (i, slot) in var_map.iter_mut().enumerate().skip(1).take(u.num_variables - 1) {
        let left_entry = u.old_left[i].and_then(|ov| left.var_map[ov as usize]);
        let right_entry = u.old_right[i].and_then(|ov| right.var_map[ov as usize]);
        // Both -> Variable.__add__ (filled True); exactly one -> as_filled(True).
        if left_entry.is_some() || right_entry.is_some() {
            *slot = Some(true);
        }
    }
    Rc::new(Node {
        rule,
        cat,
        left: Some(left),
        right: Some(right),
        var_map,
        score: Cell::new(0.0),
        word: None,
    })
}

// ---------------------------------------------------------------------------
// Unify (tree.py:113-204)
// ---------------------------------------------------------------------------

pub struct Unify<'a> {
    pub feature: u8,
    pub num_variables: usize,

    pub trans_left: [Option<u8>; VAR_SLOTS],
    pub trans_right: [Option<u8>; VAR_SLOTS],
    pub old_left: [Option<u8>; VAR_SLOTS],
    pub old_right: [Option<u8>; VAR_SLOTS],

    left: &'a Node,
    right: &'a Node,
    result_is_left: bool,

    pub arg: CatRef,
    pub res: CatRef,
}

impl<'a> Unify<'a> {
    pub fn new(left: &'a Node, right: &'a Node, result_is_left: bool) -> Self {
        let (arg, res) = if result_is_left {
            (right.cat.clone(), left.cat.clone())
        } else {
            (left.cat.clone(), right.cat.clone())
        };
        Unify {
            feature: FEATURE_NONE,
            num_variables: 1,
            trans_left: [None; VAR_SLOTS],
            trans_right: [None; VAR_SLOTS],
            old_left: [None; VAR_SLOTS],
            old_right: [None; VAR_SLOTS],
            left,
            right,
            result_is_left,
            arg,
            res,
        }
    }

    /// tree.py:136-148.
    pub fn unify(&mut self, arg_part: &Category, res_part: &Category) -> bool {
        let (left_cat, right_cat) = if self.result_is_left {
            (res_part, arg_part)
        } else {
            (arg_part, res_part)
        };

        if !self.unify_recursive(left_cat, right_cat) {
            return false;
        }

        // arg FIRST, res SECOND, regardless of orientation.
        let arg = self.arg.clone();
        let res = self.res.clone();
        let arg_is_left = !self.result_is_left;
        self.add_vars(&arg, arg_is_left);
        self.add_vars(&res, self.result_is_left);

        true
    }

    /// tree.py:150-186.
    fn unify_recursive(&mut self, left: &Category, right: &Category) -> bool {
        if left.is_atomic() {
            if left.atom != right.atom {
                return false;
            }
            if left.atom == ATOM_S {
                if left.feature == FEATURE_X {
                    self.feature = right.feature;
                } else if right.feature == FEATURE_X {
                    self.feature = left.feature;
                } else if left.feature != right.feature {
                    return false;
                }
            }
        } else if !(left.dir == right.dir
            && self.unify_recursive(
                left.result_ref().unwrap(),
                right.result_ref().unwrap(),
            )
            && self.unify_recursive(
                left.argument_ref().unwrap(),
                right.argument_ref().unwrap(),
            ))
        {
            return false;
        }

        if self.trans_left[left.var as usize].is_none()
            && self.trans_right[right.var as usize].is_none()
        {
            // The double-filled rejection: both var_maps present & filled.
            let v1 = self.left.var_map[left.var as usize];
            let v2 = self.right.var_map[right.var as usize];
            if let (Some(true), Some(true)) = (v1, v2) {
                return false;
            }

            let n = self.num_variables as u8;
            self.trans_left[left.var as usize] = Some(n);
            self.trans_right[right.var as usize] = Some(n);
            self.old_left[self.num_variables] = Some(left.var);
            self.old_right[self.num_variables] = Some(right.var);
            self.num_variables += 1;
        }

        true
    }

    /// tree.py:188-195. Iterates variable ids in ascending order.
    fn add_vars(&mut self, cat: &Category, is_left: bool) {
        let mut vars = cat.vars;
        while vars != 0 {
            let v = vars.trailing_zeros() as usize;
            vars &= vars - 1;

            let present = if is_left {
                self.trans_left[v].is_some()
            } else {
                self.trans_right[v].is_some()
            };
            if !present {
                let n = self.num_variables as u8;
                if is_left {
                    self.trans_left[v] = Some(n);
                    self.old_left[self.num_variables] = Some(v as u8);
                } else {
                    self.trans_right[v] = Some(n);
                    self.old_right[self.num_variables] = Some(v as u8);
                }
                self.num_variables += 1;
            }
        }
    }

    /// tree.py:197-198. trans_left.get(left.cat.var, 0).
    pub fn get_new_outer_var(&self) -> u8 {
        self.trans_left[self.left.cat.var as usize].unwrap_or(0)
    }

    fn trans_arg(&self) -> &[Option<u8>; VAR_SLOTS] {
        if self.result_is_left {
            &self.trans_right
        } else {
            &self.trans_left
        }
    }

    fn trans_res(&self) -> &[Option<u8>; VAR_SLOTS] {
        if self.result_is_left {
            &self.trans_left
        } else {
            &self.trans_right
        }
    }

    /// tree.py:200-201.
    pub fn translate_arg(&self, cat: &Category) -> CatRef {
        cat.translate(self.trans_arg(), self.feature)
    }

    /// tree.py:203-204.
    pub fn translate_res(&self, cat: &Category) -> CatRef {
        cat.translate(self.trans_res(), self.feature)
    }

    /// trans_arg[var] (used by generalised composition for inner vars).
    pub fn trans_arg_var(&self, var: u8) -> u8 {
        self.trans_arg()[var as usize].expect("trans_arg variable missing")
    }
}
