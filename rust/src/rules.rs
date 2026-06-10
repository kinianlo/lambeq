// Port of lambeq/bobcat/rules.py core combinators.

use std::rc::Rc;

use crate::category::{
    self, CatRef, ATOM_COMMA, ATOM_CONJ, ATOM_N, ATOM_NP, ATOM_SEMICOLON, FEATURE_X,
};
use crate::grammar::Grammar;
use crate::tree::{
    adjectival_conj_tree, binary_combinator, coordination, left_punct_tree,
    right_punct_tree, Node, Rule, Unify,
};

// ---------------------------------------------------------------------------
// CatKind (rules.py:32-63)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CatKind {
    Atom,
    Backward,
    Forward,
    Conj,
    Punct,
}

impl CatKind {
    pub fn of(cat: &category::Category) -> CatKind {
        if cat.atom_is_punct() {
            CatKind::Punct
        } else if cat.atom == ATOM_CONJ {
            CatKind::Conj
        } else if cat.bwd() {
            CatKind::Backward
        } else if cat.fwd() {
            CatKind::Forward
        } else {
            CatKind::Atom
        }
    }

    pub fn is_standard(self) -> bool {
        matches!(self, CatKind::Atom | CatKind::Backward | CatKind::Forward)
    }
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

pub struct Rules {
    pub eisner_normal_form: bool,
    pub grammar: Grammar,
    // Pattern categories parsed at construction (rules.py uses Category.parse).
    pat_s_np: CatRef,
    pat_sdcl_sdcl: CatRef,
}

impl Rules {
    pub fn new(eisner_normal_form: bool, grammar: Grammar) -> Rules {
        Rules {
            eisner_normal_form,
            grammar,
            pat_s_np: category::parse(r"S\NP", '+'),
            pat_sdcl_sdcl: category::parse(r"S[dcl]\S[dcl]", '+'),
        }
    }

    /// rules.py:126-169.
    pub fn combine(&self, left: &Rc<Node>, right: &Rc<Node>) -> Vec<Rc<Node>> {
        if !self.grammar.has_rule_instance(&left.cat, &right.cat) {
            return Vec::new();
        }

        let lk = CatKind::of(&left.cat);
        let rk = CatKind::of(&right.cat);

        let mut results: Vec<Rc<Node>> = Vec::new();

        if lk == CatKind::Atom && rk == CatKind::Backward {
            if let Some(n) = self.backward_application(left, right) { results.push(n) }
        } else if lk == CatKind::Forward
            && (rk == CatKind::Atom || rk == CatKind::Conj)
        {
            if let Some(n) = self.forward_application(left, right) { results.push(n) }
        } else if lk == CatKind::Forward && rk == CatKind::Forward {
            let mut res = self.forward_application(left, right);
            if res.is_none() {
                res = self.forward_composition(left, right);
            }
            if let Some(n) = res { results.push(n) }
        } else if lk == CatKind::Forward && rk == CatKind::Backward {
            let mut res = self.backward_application(left, right);
            if res.is_none() {
                res = self.forward_application(left, right);
                if res.is_none() {
                    res = self.backward_cross_composition(left, right);
                }
            }
            if let Some(n) = res { results.push(n) }
        } else if lk == CatKind::Backward && rk == CatKind::Backward {
            let mut res = self.backward_application(left, right);
            if res.is_none() {
                res = self.backward_composition(left, right);
            }
            if let Some(n) = res { results.push(n) }
        } else if lk == CatKind::Conj && rk == CatKind::Atom {
            if let Some(n) = self.coordination(left, right) { results.push(n) }
            if let Some(n) = self.adjectival_conj(left, right) { results.push(n) }
        } else if lk == CatKind::Conj
            && (rk == CatKind::Backward || rk == CatKind::Forward)
        {
            let mut res = self.backward_application(left, right);
            if res.is_none() {
                res = self.coordination(left, right);
            }
            if let Some(n) = res { results.push(n) }
        } else if lk == CatKind::Punct && rk.is_standard() {
            results.extend(self.left_punct(left, right));
        } else if lk.is_standard() && rk == CatKind::Punct {
            results.extend(self.right_punct(left, right));
        }

        results
    }

    // -- punctuation (rules.py:202-246), type-changing arms omitted (Task 3) --

    fn left_punct(&self, left: &Rc<Node>, right: &Rc<Node>) -> Vec<Rc<Node>> {
        let mut results = Vec::new();
        if !right.coordinated_or_type_raised() {
            results.push(left_punct_tree(left.clone(), right.clone()));
        }

        // left punct coordination
        let la = left.cat.atom;
        if (la == ATOM_COMMA || la == ATOM_SEMICOLON)
            && !right.coordinated_or_type_raised()
            && !right.cat.atom_is_punct()
        {
            let cat = category::Category::slash(
                right.cat.clone(),
                b'\\',
                right.cat.clone(),
                0,
                false,
                0,
            );
            results.push(coordination(cat, left.clone(), right.clone()));
        }
        results
    }

    fn right_punct(&self, left: &Rc<Node>, right: &Rc<Node>) -> Vec<Rc<Node>> {
        let mut results = Vec::new();
        if !left.coordinated_or_type_raised() {
            results.push(right_punct_tree(left.clone(), right.clone()));
        }
        results
    }

    // -- application (rules.py:248-266, 318-328) --

    fn backward_application(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
    ) -> Option<Rc<Node>> {
        if right.cat.bwd()
            && !left.coordinated_or_type_raised()
            && !(self.eisner_normal_form && right.bwd_comp())
        {
            self.application(left, right, false)
        } else {
            None
        }
    }

    fn forward_application(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
    ) -> Option<Rc<Node>> {
        if left.cat.fwd()
            && !right.coordinated_or_type_raised()
            && !(self.eisner_normal_form && left.fwd_comp())
        {
            self.application(left, right, true)
        } else {
            None
        }
    }

    fn application(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
        fwd: bool,
    ) -> Option<Rc<Node>> {
        let mut u = Unify::new(left, right, fwd);
        let arg = u.arg.clone();
        let res = u.res.clone();
        let res_arg = res.argument_ref()?;
        if u.unify(&arg, res_arg) {
            let res_result = res.result_ref()?;
            let result = u.translate_res(res_result);
            let rule = if fwd { Rule::FA } else { Rule::BA };
            Some(binary_combinator(rule, result, left.clone(), right.clone(), &u))
        } else {
            None
        }
    }

    // -- coordination (rules.py:301-316) --

    fn coordination(&self, left: &Rc<Node>, right: &Rc<Node>) -> Option<Rc<Node>> {
        if left.cat.atom == ATOM_CONJ && !right.coordinated_or_type_raised() {
            let cat = category::Category::slash(
                right.cat.clone(),
                b'\\',
                right.cat.clone(),
                0,
                false,
                0,
            );
            Some(coordination(cat, left.clone(), right.clone()))
        } else {
            None
        }
    }

    fn adjectival_conj(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
    ) -> Option<Rc<Node>> {
        if left.cat.atom == ATOM_CONJ && right.cat.atom == ATOM_N {
            Some(adjectival_conj_tree(left.clone(), right.clone()))
        } else {
            None
        }
    }

    // -- composition (rules.py:268-360) --

    fn backward_composition(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
    ) -> Option<Rc<Node>> {
        if left.cat.bwd()
            && right.cat.bwd()
            && !left.coordinated()
            && !right.coordinated()
            && !(self.eisner_normal_form && right.bwd_comp())
        {
            self.composition(left, right, Comp::Bc)
        } else {
            None
        }
    }

    fn forward_composition(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
    ) -> Option<Rc<Node>> {
        if left.cat.fwd()
            && right.cat.fwd()
            && !(self.eisner_normal_form && left.fwd_comp())
        {
            self.composition(left, right, Comp::Fc)
        } else {
            None
        }
    }

    fn backward_cross_composition(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
    ) -> Option<Rc<Node>> {
        if left.cat.fwd()
            && right.cat.bwd()
            && !right.coordinated()
            && right
                .cat
                .argument_ref()
                .map(|a| a.atom != ATOM_N && a.atom != ATOM_NP)
                .unwrap_or(false)
        {
            self.composition(left, right, Comp::Bx)
        } else {
            None
        }
    }

    fn composition(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
        comp: Comp,
    ) -> Option<Rc<Node>> {
        let mut u = Unify::new(left, right, comp == Comp::Fc);
        let arg = u.arg.clone();
        let res = u.res.clone();

        let arg_result = arg.result_ref();
        let res_arg = res.argument_ref();

        let unified = match (arg_result, res_arg) {
            (Some(ar), Some(ra)) => u.unify(ar, ra),
            // Python would raise AttributeError here -> caught downstream;
            // the generalised fallbacks return None for these shapes anyway.
            _ => false,
        };

        if !unified {
            return match comp {
                Comp::Fc => self.generalised_forward_composition(left, right),
                Comp::Bx => self.generalised_backward_cross_composition(left, right),
                Comp::Bc => self.generalised_backward_composition(left, right),
            };
        }

        let result_cat = u.translate_res(res.result_ref()?);
        let arg_cat = u.translate_arg(arg.argument_ref()?);
        let var = u.get_new_outer_var();
        let dir = if comp == Comp::Bc { b'\\' } else { b'/' };
        let new_cat = category::Category::slash(result_cat, dir, arg_cat, var, false, 0);

        let rule = match comp {
            Comp::Bc => Rule::BC,
            Comp::Fc => Rule::FC,
            Comp::Bx => Rule::BX,
        };
        Some(binary_combinator(rule, new_cat, left.clone(), right.clone(), &u))
    }

    // -- generalised composition (rules.py:362-485) --

    fn gc2(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
        comp: Comp,
    ) -> Option<Rc<Node>> {
        let mut u = Unify::new(left, right, comp == Comp::Fc);
        let arg = u.arg.clone();
        let res = u.res.clone();

        let arg_result = arg.result_ref()?;
        let arg_result_result = arg_result.result_ref()?;
        let res_arg = res.argument_ref()?;
        if !u.unify(arg_result_result, res_arg) {
            return None;
        }

        let inner_result = u.translate_res(res.result_ref()?);
        let inner_argument = u.translate_arg(arg_result.argument_ref()?);
        let inner_var = u.trans_arg_var(arg_result.var);
        let new_result = category::Category::slash(
            inner_result,
            arg_result.dir,
            inner_argument,
            inner_var,
            arg_result.has_relation,
            0,
        );
        let new_argument = u.translate_arg(arg.argument_ref()?);
        let var = u.get_new_outer_var();
        let new_category = category::Category::slash(
            new_result,
            arg.dir,
            new_argument,
            var,
            arg.has_relation,
            0,
        );
        let rule = if comp == Comp::Fc { Rule::GFC } else { Rule::GBX };
        Some(binary_combinator(rule, new_category, left.clone(), right.clone(), &u))
    }

    fn gc3(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
        comp: Comp,
    ) -> Option<Rc<Node>> {
        let mut u = Unify::new(left, right, comp == Comp::Fc);
        let arg = u.arg.clone();
        let res = u.res.clone();

        let arg_result = arg.result_ref()?;
        let arg_result_result = arg_result.result_ref()?;
        let arg_result_result_result = arg_result_result.result_ref()?;
        let res_arg = res.argument_ref()?;
        if !u.unify(arg_result_result_result, res_arg) {
            return None;
        }

        let inner_inner_result = u.translate_res(res.result_ref()?);
        let inner_inner_argument =
            u.translate_arg(arg_result_result.argument_ref()?);
        let inner_inner_var = u.trans_arg_var(arg_result_result.var);
        let inner_result = category::Category::slash(
            inner_inner_result,
            arg_result_result.dir,
            inner_inner_argument,
            inner_inner_var,
            arg_result_result.has_relation,
            0,
        );

        let inner_argument = u.translate_arg(arg_result.argument_ref()?);
        let inner_var = u.trans_arg_var(arg_result.var);

        let new_result = category::Category::slash(
            inner_result,
            arg_result.dir,
            inner_argument,
            inner_var,
            arg_result.has_relation,
            0,
        );
        let new_argument = u.translate_arg(arg.argument_ref()?);

        let var = u.get_new_outer_var();
        let new_cat = category::Category::slash(
            new_result,
            arg.dir,
            new_argument,
            var,
            arg.has_relation,
            0,
        );
        let rule = if comp == Comp::Fc { Rule::GFC } else { Rule::GBX };
        Some(binary_combinator(rule, new_cat, left.clone(), right.clone(), &u))
    }

    fn generalised_forward_composition(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
    ) -> Option<Rc<Node>> {
        // left.cat.argument
        let la = left.cat.argument_ref()?;
        if !self.pat_s_np.matches(la) {
            return None;
        }
        // right.cat.result
        let rcr = right.cat.result_ref()?;
        if !rcr.fwd() {
            return None;
        }
        // right.cat.result.result.result.feature
        let rcrr = rcr.result_ref()?;
        let rcrrr = rcrr.result_ref()?;
        if rcrrr.feature == FEATURE_X {
            return None;
        }

        if let Some(res) = self.gc2(left, right, Comp::Fc) {
            return Some(res);
        }

        // feat = right.cat.result.result.result.result.feature
        let rcrrrr = rcrrr.result_ref()?;
        let feat = rcrrrr.feature;
        if rcrr.fwd() && feat != FEATURE_X {
            return self.gc3(left, right, Comp::Fc);
        }
        None
    }

    fn generalised_backward_composition(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
    ) -> Option<Rc<Node>> {
        // Category.parse('S[dcl]\\S[dcl]').matches(left.cat.result)
        let lcr = match left.cat.result_ref() {
            Some(c) if self.pat_sdcl_sdcl.matches(c) => c,
            _ => return None,
        };
        // left.var_map[left.cat.result.result.var].filled
        let lcrr = lcr.result_ref()?;
        match left.var_map[lcrr.var as usize] {
            Some(filled) => {
                if !filled {
                    return None;
                }
            }
            None => return None,
        }

        let mut u = Unify::new(left, right, false);
        let arg = u.arg.clone();
        let res = u.res.clone();
        let arg_rr = arg.result_ref()?.result_ref()?;
        let res_arg = res.argument_ref()?;
        if !u.unify(arg_rr, res_arg) {
            return None;
        }
        Some(binary_combinator(
            Rule::GBC,
            left.cat.clone(),
            left.clone(),
            right.clone(),
            &u,
        ))
    }

    fn generalised_backward_cross_composition(
        &self,
        left: &Rc<Node>,
        right: &Rc<Node>,
    ) -> Option<Rc<Node>> {
        // Category.parse('S\NP').matches(right.cat.argument)
        let ra = right.cat.argument_ref()?;
        if !self.pat_s_np.matches(ra) {
            return None;
        }
        let lcr = left.cat.result_ref()?;
        if !lcr.fwd() {
            return None;
        }
        // left.cat.result.result.result.feature
        let lcrr = lcr.result_ref()?;
        let lcrrr = lcrr.result_ref()?;
        if lcrrr.feature == FEATURE_X {
            return None;
        }

        if let Some(res) = self.gc2(left, right, Comp::Bx) {
            return Some(res);
        }

        let lcrrrr = lcrrr.result_ref()?;
        let feat = lcrrrr.feature;
        if lcrr.fwd() && feat != FEATURE_X {
            return self.gc3(left, right, Comp::Bx);
        }
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Comp {
    Bc,
    Fc,
    Bx,
}
