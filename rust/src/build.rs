//! Rust port of lambeq/backend/fast/build.py + the _to_fast_diagram
//! stack-machine assembly. Combinators take FTy and return FDiagram.

use crate::fdiagram::{FBox, FDiagram, FTy, CAP, CUP, SWAP};

fn cup_box(a: u32, b: u32) -> FBox {
    FBox { name: "CUP".into(), dom: FTy(vec![a, b]), cod: FTy(vec![]),
           kind: CUP, z: 0, is_dagger: false }
}
fn cap_box(a: u32, b: u32) -> FBox {
    FBox { name: "CAP".into(), dom: FTy(vec![]), cod: FTy(vec![a, b]),
           kind: CAP, z: 0, is_dagger: false }
}
fn swap_box(a: u32, b: u32) -> FBox {
    FBox { name: "SWAP".into(), dom: FTy(vec![a, b]),
           cod: FTy(vec![b, a]), kind: SWAP, z: 0, is_dagger: false }
}

fn id(ty: &FTy) -> FDiagram {
    FDiagram::id(ty)
}

/// Nested cups, innermost first (matches fast.diagram.cups).
pub fn cups(left: &FTy, right: &FTy) -> FDiagram {
    let n = left.len();
    let mut terms = Vec::with_capacity(n);
    for i in (0..n).rev() {
        terms.push((cup_box(left.0[i], right.0[n - 1 - i]), i as u32));
    }
    FDiagram { dom: left.tensor(right), terms, cod: FTy(vec![]) }
}

/// Decomposed complex swap (mirrors fast.build.swaps / grammar.Swap).
pub fn swaps(left: &FTy, right: &FTy) -> FDiagram {
    let nl = left.len();
    let mut frontier: Vec<u32> = left.0.clone();
    frontier.extend_from_slice(&right.0);
    let mut terms = Vec::new();
    for start in 0..right.len() {
        for i in (0..nl).rev() {
            let off = start + i;
            let (a, b) = (frontier[off], frontier[off + 1]);
            terms.push((swap_box(a, b), off as u32));
            frontier.swap(off, off + 1);
        }
    }
    FDiagram { dom: left.tensor(right), terms, cod: right.tensor(left) }
}

pub fn fa(left: &FTy, right: &FTy) -> FDiagram {
    id(left).tensor(&cups(&right.l(), right))
}
pub fn ba(left: &FTy, right: &FTy) -> FDiagram {
    cups(left, &left.r()).tensor(&id(right))
}
pub fn fc(left: &FTy, mid: &FTy, right: &FTy) -> FDiagram {
    id(left).tensor(&cups(&mid.l(), mid)).tensor(&id(&right.l()))
}
pub fn bc(left: &FTy, mid: &FTy, right: &FTy) -> FDiagram {
    id(&left.r()).tensor(&cups(mid, &mid.r())).tensor(&id(right))
}
pub fn fx(left: &FTy, mid: &FTy, right: &FTy) -> FDiagram {
    let top = id(left).tensor(&swaps(&mid.l(), &right.r())).tensor(&id(mid));
    let bot = swaps(left, &right.r()).tensor(&cups(&mid.l(), mid));
    top.then(&bot)
}
pub fn bx(left: &FTy, mid: &FTy, right: &FTy) -> FDiagram {
    let top = id(mid).tensor(&swaps(&left.l(), &mid.r())).tensor(&id(right));
    let bot = cups(mid, &mid.r()).tensor(&swaps(&left.l(), right));
    top.then(&bot)
}
pub fn gfc(left: &FTy, mid: &FTy, tail: &FTy) -> FDiagram {
    id(left).tensor(&cups(&mid.l(), mid)).tensor(&id(tail))
}
pub fn gbc(prefix: &FTy, mid: &FTy, right: &FTy) -> FDiagram {
    id(prefix).tensor(&cups(mid, &mid.r())).tensor(&id(right))
}
pub fn gfx(mid: &FTy, l: &FTy, join: &FTy, r: &FTy) -> FDiagram {
    let inner_top = swaps(&mid.tensor(&join.l()), l).tensor(&id(join));
    let inner_bot = id(&l.tensor(mid)).tensor(&cups(&join.l(), join));
    inner_top.then(&inner_bot).tensor(&id(r))
}
pub fn gbx(mid: &FTy, l: &FTy, join: &FTy, r: &FTy) -> FDiagram {
    let inner_top = id(join).tensor(&swaps(r, &join.r().tensor(mid)));
    let inner_bot = cups(join, &join.r()).tensor(&id(&mid.tensor(r)));
    id(l).tensor(&inner_top.then(&inner_bot))
}

/// Left-adjoint caps for type-raising: right == left.l per atom. Builds
/// the CAP boxes directly (no right-adjoint validation), mirroring
/// build.py `_caps_any`.
fn caps_any(left: &FTy, right: &FTy) -> FDiagram {
    let n = left.len();
    let mut terms = Vec::with_capacity(n);
    for i in 0..n {
        terms.push((cap_box(left.0[i], right.0[n - 1 - i]), i as u32));
    }
    FDiagram { dom: FTy(vec![]), terms, cod: left.tensor(right) }
}
pub fn ftr(result: &FTy, dom0: &FTy) -> FDiagram {
    caps_any(result, &result.l()).tensor(&id(dom0))
}
pub fn btr(result: &FTy, dom0: &FTy) -> FDiagram {
    id(dom0).tensor(&caps_any(&result.r(), result))
}

/// RULE dispatch: args already extracted by the Python emitter, in the
/// order each combinator expects. Tags per the shared constant table.
pub fn rule_layer(tag: u8, types: &[FTy]) -> FDiagram {
    match tag {
        0 => fa(&types[0], &types[1]),
        1 => ba(&types[0], &types[1]),
        2 => fc(&types[0], &types[1], &types[2]),
        3 => bc(&types[0], &types[1], &types[2]),
        4 => fx(&types[0], &types[1], &types[2]),
        5 => bx(&types[0], &types[1], &types[2]),
        6 => gfc(&types[0], &types[1], &types[2]),
        7 => gbc(&types[0], &types[1], &types[2]),
        8 => gfx(&types[0], &types[1], &types[2], &types[3]),
        9 => gbx(&types[0], &types[1], &types[2], &types[3]),
        10 => ftr(&types[0], &types[1]),
        11 => btr(&types[0], &types[1]),
        12 => id(&types[0]), // LP: id(right)
        13 => id(&types[0]), // RP: id(left)
        _ => panic!("unknown rule tag {tag}"),
    }
}

/// One build-program node (decoded from PyO3 in Task 3).
pub struct Node {
    pub node_type: u8,
    pub rule_tag: u8,
    pub arity: u8,
    pub types: Vec<FTy>,
    pub name: String,
}

/// Stack-machine assembly mirroring _to_fast_diagram. Stack holds
/// (words, grammar) FDiagram pairs.
pub fn assemble(nodes: &[Node]) -> FDiagram {
    let mut stack: Vec<(FDiagram, FDiagram)> = Vec::new();
    for nd in nodes {
        match nd.node_type {
            0 => {
                // WORD
                let cod = &nd.types[0];
                let wbox = FBox { name: nd.name.clone(),
                                  dom: FTy(vec![]), cod: cod.clone(),
                                  kind: crate::fdiagram::WORD, z: 0,
                                  is_dagger: false };
                let words = FDiagram { dom: FTy(vec![]),
                                       terms: vec![(wbox, 0)],
                                       cod: cod.clone() };
                stack.push((words, id(cod)));
            }
            1 => {
                // PUNC
                let e = FDiagram::id(&FTy(vec![]));
                stack.push((e.clone(), e));
            }
            2 => {
                // UNARY_SWAP: types = [right, left]
                let (cw, cg) = stack.pop().expect("build::assemble: stack underflow");
                let layer = swaps(&nd.types[0], &nd.types[1]);
                stack.push((cw, cg.then(&layer)));
            }
            3 => {
                // RULE
                let k = nd.arity as usize;
                let mut kids: Vec<(FDiagram, FDiagram)> = Vec::with_capacity(k);
                for _ in 0..k {
                    kids.push(stack.pop().expect("build::assemble: stack underflow"));
                }
                kids.reverse(); // restore left-to-right order
                let mut words = FDiagram::id(&FTy(vec![]));
                let mut diag = FDiagram::id(&FTy(vec![]));
                for (w, d) in &kids {
                    words = words.tensor(w);
                    diag = diag.tensor(d);
                }
                let layer = rule_layer(nd.rule_tag, &nd.types);
                diag = diag.then(&layer);
                stack.push((words, diag));
            }
            _ => panic!("unknown node type {}", nd.node_type),
        }
    }
    let (words, diag) = stack.pop().expect("build::assemble: stack underflow");
    words.then(&diag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdiagram::atom;

    fn n() -> FTy { FTy(vec![atom("n", 0)]) }
    fn s() -> FTy { FTy(vec![atom("s", 0)]) }

    #[test]
    fn fa_shape() {
        let d = fa(&n(), &s());
        assert!(d.validate().is_ok());
        assert_eq!(d.cod, n());
    }

    #[test]
    fn swaps_shape() {
        let left = n().tensor(&s());
        let right = FTy(vec![atom("p", 0)]);
        let d = swaps(&left, &right);
        assert!(d.validate().is_ok());
        assert_eq!(d.cod, right.tensor(&left));
        assert!(d.terms.iter().all(|(b, _)| b.kind == SWAP));
    }

    #[test]
    fn assemble_punctuation_rp() {
        // word(n) + punc, RP drops the punc, keeping n.
        let prog = vec![
            Node { node_type: 0, rule_tag: 0, arity: 0,
                   types: vec![n()], name: "w".into() },
            Node { node_type: 1, rule_tag: 0, arity: 0,
                   types: vec![], name: String::new() },
            Node { node_type: 3, rule_tag: 13, arity: 2,
                   types: vec![n()], name: String::new() },
        ];
        let d = assemble(&prog);
        assert!(d.validate().is_ok());
        assert_eq!(d.cod, n());
        // exactly one term: the word box (punc dropped, RP is identity)
        assert_eq!(d.terms.len(), 1);
    }

    #[test]
    fn assemble_unary_swap() {
        // word(n@s) then a unary swap (right=n, left=s) -> cod s@n.
        let ns = n().tensor(&s());
        let prog = vec![
            Node { node_type: 0, rule_tag: 0, arity: 0,
                   types: vec![ns.clone()], name: "w".into() },
            Node { node_type: 2, rule_tag: 0, arity: 1,
                   types: vec![n(), s()], name: String::new() },
        ];
        let d = assemble(&prog);
        assert!(d.validate().is_ok());
        assert_eq!(d.cod, s().tensor(&n()));
    }

    #[test]
    fn assemble_two_words_fa() {
        let xl = s().tensor(&n().l()); // s @ n.l  (= X/Y)
        let prog = vec![
            Node { node_type: 0, rule_tag: 0, arity: 0,
                   types: vec![xl.clone()], name: "w1".into() },
            Node { node_type: 0, rule_tag: 0, arity: 0,
                   types: vec![n()], name: "w2".into() },
            Node { node_type: 3, rule_tag: 0, arity: 2,
                   types: vec![s(), n()], name: String::new() },
        ];
        let d = assemble(&prog);
        assert!(d.validate().is_ok());
        assert_eq!(d.cod, s());
        assert_eq!(d.dom, FTy(vec![]));
    }
}
