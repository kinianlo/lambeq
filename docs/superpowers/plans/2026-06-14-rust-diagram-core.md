# Rust Diagram Core + Construction Kernel (Stage A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust-native diagram data model + a batched, rayon-parallel construction kernel in the `bobcat_rs` crate that builds `FDiagram`s from Python-resolved CCG trees, bit-identical to `to_diagram` and faster than the Python direct path.

**Architecture:** Python keeps `_resolved()` + `CCGType` accessor extraction and emits a flat post-order "build program" per tree (rule tags + pregroup-type args as `(name, z)` lists). Rust executes the programs (intern atoms → build combinator fragments → stack-machine assembly) in parallel, returning `RsDiagram` handles. `RsDiagram.export()` + a Python materializer rebuild the existing Python `FDiagram` for the gates and drop-in compat. Stages B/C will later consume `RsDiagram` natively.

**Tech Stack:** Rust (PyO3 0.22 abi3-py310, rayon) in `rust/`; Python 3.10+. Oracles: `lambeq/backend/fast/build.py` (already gated vs grammar) is the exact reference for the Rust combinators; `CCGTree.to_diagram` and `CCGTree.to_fast_diagram(backend='python')` are the differential oracles.

**Spec:** `docs/superpowers/specs/2026-06-14-rust-diagram-core-design.md`

---

## Conventions (every task)

- Repo `/home/kinianlo/projects/lambeq`, branch `rust-diagram-core`.
  `PY=~/.pyenv/versions/qnlp/bin/python`. Rust: `cargo` (1.96, on PATH);
  build the extension with `~/.pyenv/versions/qnlp/bin/maturin develop
  --release -m rust/Cargo.toml` OR `$PY -m pip install ./rust` (rebuilds
  `bobcat_rs`). 600000ms timeouts on anything loading the Bobcat model or
  doing a release Rust build.
- Rust unit tests: `cargo test --manifest-path rust/Cargo.toml`.
- Commit style: plain sentence + trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- The Python fast core (`lambeq/backend/fast/`) data model is NOT changed.
  `to_diagram` and `CCGTree.to_fast_diagram(backend='python')` (the
  current default) are untouched oracles. The Rust path is OPT-IN.

## Kind + rule-tag constants (shared Python/Rust — must agree exactly)

```
FBox kind:   PLAIN=0  WORD=1  CUP=2  CAP=3  SWAP=4  SPIDER=5
Node type:   WORD=0   PUNC=1  UNARY_SWAP=2  RULE=3
Rule tag:    FA=0  BA=1  FC=2  BC=3  FX=4  BX=5  GFC=6  GBC=7
             GFX=8 GBX=9 FTR=10 BTR=11 LP=12 RP=13
```

## Build-program format (the boundary contract)

One tree → a flat **post-order** list of nodes. Each node is the PyO3
tuple `(node_type: u8, rule_tag: u8, arity: u8, types: list[list[(str,
int)]], text: str)`:
- WORD: `(0, 0, 0, [cod_atoms], text)` — `cod_atoms` = list of `(name, z)`.
- PUNC: `(1, 0, 0, [], "")`.
- UNARY_SWAP: `(2, 0, 1, [right_atoms, left_atoms], "")`.
- RULE: `(3, rule_tag, arity, [arg_atoms...], "")` — `arity` is 1 (type-
  raising) or 2 (binary); `types` are the pregroup args in the order the
  Rust `build_<rule>` expects (FA: `[left.result, right]`; BA: `[left,
  right.result]`; FC/BC: `[l.left, l.right, r.right]`; FX: `[l.left,
  l.right, r.left]`; BX: `[l.right, l.left, r.right]`; GFC: `[l.result,
  mid, right_tail]`; GBC: `[left_prefix, mid, r.result]`; GFX/GBX: `[mid,
  l, join, r]`; FTR/BTR: `[result, dom0]`; LP: `[right]`; RP: `[left]`).

A batch = `list[program]`. `bobcat_rs.build_diagrams(batch)` →
`list[RsDiagram]`.

---

### Task 1: Rust data model (`rust/src/fdiagram.rs`)

**Files:** Create `rust/src/fdiagram.rs`; modify `rust/src/lib.rs` (add
`mod fdiagram;`).

- [ ] **Step 1: implement `rust/src/fdiagram.rs`:**

```rust
//! Fast-diagram data model in Rust: interned atoms, FTy, FBox, FDiagram.
//! Mirrors lambeq/backend/fast/{types,diagram}.py. Atom ids are local to
//! this Rust intern table and are NOT the Python ids (materialisation
//! re-interns by (name, z) on the Python side).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub const PLAIN: u8 = 0;
pub const WORD: u8 = 1;
pub const CUP: u8 = 2;
pub const CAP: u8 = 3;
pub const SWAP: u8 = 4;
pub const SPIDER: u8 = 5;

struct Intern {
    ids: HashMap<(String, i32), u32>,
    names: Vec<String>,
    zs: Vec<i32>,
    l: Vec<i32>, // atom id -> left-adjoint id, -1 = not built
    r: Vec<i32>,
}

fn intern() -> &'static Mutex<Intern> {
    static I: OnceLock<Mutex<Intern>> = OnceLock::new();
    I.get_or_init(|| {
        Mutex::new(Intern {
            ids: HashMap::new(),
            names: Vec::new(),
            zs: Vec::new(),
            l: Vec::new(),
            r: Vec::new(),
        })
    })
}

pub fn atom(name: &str, z: i32) -> u32 {
    let mut g = intern().lock().unwrap();
    if let Some(&id) = g.ids.get(&(name.to_string(), z)) {
        return id;
    }
    let id = g.names.len() as u32;
    g.ids.insert((name.to_string(), z), id);
    g.names.push(name.to_string());
    g.zs.push(z);
    g.l.push(-1);
    g.r.push(-1);
    id
}

pub fn atom_name(i: u32) -> String {
    intern().lock().unwrap().names[i as usize].clone()
}

pub fn atom_z(i: u32) -> i32 {
    intern().lock().unwrap().zs[i as usize]
}

// NOTE: never call atom() while holding the lock (std Mutex is not
// reentrant). Read (name, z) under the lock, release, intern, re-lock.
pub fn atom_l(i: u32) -> u32 {
    {
        let g = intern().lock().unwrap();
        let j = g.l[i as usize];
        if j >= 0 {
            return j as u32;
        }
    }
    let (name, z) = {
        let g = intern().lock().unwrap();
        (g.names[i as usize].clone(), g.zs[i as usize])
    };
    let j = atom(&name, z - 1);
    let mut g = intern().lock().unwrap();
    g.l[i as usize] = j as i32;
    g.r[j as usize] = i as i32;
    j
}

pub fn atom_r(i: u32) -> u32 {
    {
        let g = intern().lock().unwrap();
        let j = g.r[i as usize];
        if j >= 0 {
            return j as u32;
        }
    }
    let (name, z) = {
        let g = intern().lock().unwrap();
        (g.names[i as usize].clone(), g.zs[i as usize])
    };
    let j = atom(&name, z + 1);
    let mut g = intern().lock().unwrap();
    g.r[i as usize] = j as i32;
    g.l[j as usize] = i as i32;
    j
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FTy(pub Vec<u32>);

impl FTy {
    pub fn empty() -> FTy {
        FTy(Vec::new())
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn tensor(&self, other: &FTy) -> FTy {
        let mut v = self.0.clone();
        v.extend_from_slice(&other.0);
        FTy(v)
    }
    pub fn l(&self) -> FTy {
        FTy(self.0.iter().rev().map(|&a| atom_l(a)).collect())
    }
    pub fn r(&self) -> FTy {
        FTy(self.0.iter().rev().map(|&a| atom_r(a)).collect())
    }
    pub fn slice(&self, lo: usize, hi: usize) -> FTy {
        FTy(self.0[lo..hi].to_vec())
    }
}

#[derive(Clone, Debug)]
pub struct FBox {
    pub name: String,
    pub dom: FTy,
    pub cod: FTy,
    pub kind: u8,
    pub z: i32,
    pub is_dagger: bool,
}

#[derive(Clone, Debug)]
pub struct FDiagram {
    pub dom: FTy,
    pub terms: Vec<(FBox, u32)>, // (box, offset)
    pub cod: FTy,
}

impl FDiagram {
    pub fn id(ty: &FTy) -> FDiagram {
        FDiagram { dom: ty.clone(), terms: Vec::new(), cod: ty.clone() }
    }

    /// self >> other (term concatenation). Assumes self.cod == other.dom
    /// (the Python build path guarantees it; validated by the gate).
    pub fn then(&self, other: &FDiagram) -> FDiagram {
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        FDiagram { dom: self.dom.clone(), terms, cod: other.cod.clone() }
    }

    /// self @ other. Other's offsets shift by len(self.cod).
    pub fn tensor(&self, other: &FDiagram) -> FDiagram {
        let shift = self.cod.len() as u32;
        let mut terms = self.terms.clone();
        terms.extend(
            other.terms.iter().map(|(b, o)| (b.clone(), o + shift)),
        );
        FDiagram {
            dom: self.dom.tensor(&other.dom),
            terms,
            cod: self.cod.tensor(&other.cod),
        }
    }

    /// Frontier replay; returns Err with a message on mismatch. Used by
    /// tests and an optional debug check, mirroring FDiagram._check.
    pub fn validate(&self) -> Result<(), String> {
        let mut frontier: Vec<u32> = self.dom.0.clone();
        for (t, (b, off)) in self.terms.iter().enumerate() {
            let off = *off as usize;
            let w = b.dom.len();
            if off + w > frontier.len() {
                return Err(format!(
                    "term {t} ({}): offset {off} width {w} exceeds frontier {}",
                    b.name,
                    frontier.len()
                ));
            }
            if frontier[off..off + w] != b.dom.0[..] {
                return Err(format!("term {t} ({}): dom mismatch", b.name));
            }
            frontier.splice(off..off + w, b.cod.0.iter().cloned());
        }
        if frontier != self.cod.0 {
            return Err("cod mismatch".to_string());
        }
        Ok(())
    }
}
```

- [ ] **Step 2: add `mod fdiagram;`** to `rust/src/lib.rs` (with the
other `mod` lines near the top).

- [ ] **Step 3: cargo unit tests** — append to `rust/src/fdiagram.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning() {
        let n = atom("n", 0);
        assert_eq!(atom("n", 0), n);
        assert_ne!(atom("n", 1), n);
        assert_eq!(atom_name(n), "n");
        assert_eq!(atom_z(n), 0);
    }

    #[test]
    fn adjoints() {
        let n = atom("n", 0);
        assert_eq!(atom_l(atom_r(n)), n);
        assert_eq!(atom_r(atom_l(n)), n);
        assert_eq!(atom_z(atom_l(n)), -1);
    }

    #[test]
    fn fty_ops() {
        let n = FTy(vec![atom("n", 0)]);
        let s = FTy(vec![atom("s", 0)]);
        let ns = n.tensor(&s);
        assert_eq!(ns.len(), 2);
        // (n @ s).l == s.l @ n.l
        assert_eq!(ns.l(), s.l().tensor(&n.l()));
        assert_eq!(ns.l().r(), ns);
    }

    #[test]
    fn diagram_tensor_then() {
        let n = FTy(vec![atom("n", 0)]);
        let f = FBox { name: "f".into(), dom: n.clone(), cod: n.clone(),
                       kind: PLAIN, z: 0, is_dagger: false };
        let fd = FDiagram { dom: n.clone(), terms: vec![(f.clone(), 0)],
                            cod: n.clone() };
        let t = fd.tensor(&fd);
        assert_eq!(t.terms.iter().map(|(_, o)| *o).collect::<Vec<_>>(),
                   vec![0, 1]);
        assert!(t.validate().is_ok());
    }
}
```

Run: `cargo test --manifest-path rust/Cargo.toml fdiagram` → all pass.

- [ ] **Step 4: commit**

```bash
git add rust/src/fdiagram.rs rust/src/lib.rs
git commit -m "Add Rust fast-diagram data model

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Rust combinators + assembly (`rust/src/build.rs`)

**Files:** Create `rust/src/build.rs`; modify `rust/src/lib.rs` (add
`mod build;`). Reference oracle: `lambeq/backend/fast/build.py` (port
1:1; gate 3 in Task 3 pins Rust ≡ Python).

- [ ] **Step 1: implement `rust/src/build.rs`:**

```rust
//! Rust port of lambeq/backend/fast/build.py + the _to_fast_diagram
//! stack-machine assembly. Combinators take FTy and return FDiagram.

use crate::fdiagram::{atom_l, atom_r, FBox, FDiagram, FTy, CAP, CUP, SWAP};

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

/// Nested caps, outermost first (matches fast.diagram.caps). Used by the
/// right-adjoint orientation; for type-raising's left-adjoint caps we
/// build directly (see ftr/btr), mirroring build.py `_caps_any`.
pub fn caps(left: &FTy, right: &FTy) -> FDiagram {
    let n = left.len();
    let mut terms = Vec::with_capacity(n);
    for i in 0..n {
        terms.push((cap_box(left.0[i], right.0[n - 1 - i]), i as u32));
    }
    FDiagram { dom: FTy(vec![]), terms, cod: left.tensor(right) }
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

/// One build-program node (decoded from PyO3).
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
                let (cw, cg) = stack.pop().unwrap();
                let layer = swaps(&nd.types[0], &nd.types[1]);
                stack.push((cw, cg.then(&layer)));
            }
            3 => {
                // RULE
                let k = nd.arity as usize;
                let mut kids: Vec<(FDiagram, FDiagram)> = Vec::with_capacity(k);
                for _ in 0..k {
                    kids.push(stack.pop().unwrap());
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
    let (words, diag) = stack.pop().unwrap();
    words.then(&diag)
}
```

- [ ] **Step 2: add `mod build;`** to `rust/src/lib.rs`.

- [ ] **Step 3: cargo tests** — append to `rust/src/build.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdiagram::atom;

    fn n() -> FTy { FTy(vec![atom("n", 0)]) }
    fn s() -> FTy { FTy(vec![atom("s", 0)]) }

    #[test]
    fn fa_shape() {
        // X/Y + Y -> X ; dom = X @ Y.l @ Y, cod = X
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
    fn assemble_two_words_fa() {
        // program: WORD(n/s.l? ) ... keep simple: two words then FA
        // word1 cod = n @ s.l (X/Y with X=s? ) -- use a minimal valid FA:
        // left.result = s, right = n ; left cod = s @ n.l, right cod = n
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
        assert_eq!(d.dom, FTy(vec![])); // words have empty dom
    }
}
```

Run: `cargo test --manifest-path rust/Cargo.toml` → all pass.

- [ ] **Step 4: commit**

```bash
git add rust/src/build.rs rust/src/lib.rs
git commit -m "Add Rust CCG combinators and stack-machine assembly

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: PyO3 surface + Python emitter + corpus gates

**Files:** Modify `rust/src/lib.rs` (RsDiagram, build_diagrams, register);
`lambeq/text2diagram/ccg_tree.py` (build-program emitter + backend
switch); `lambeq/backend/fast/convert.py` (`rs_to_fast`); Test
`tests/backend/test_rust_construct.py`.

- [ ] **Step 1: PyO3 surface in `rust/src/lib.rs`.** Add near the other
`#[pyclass]`es:

```rust
use crate::build::{assemble, Node};
use crate::fdiagram::{atom, atom_name, atom_z, FDiagram, FTy};

/// (name, z) pairs for one type.
type AtomList = Vec<(String, i32)>;
/// One program node from Python: (node_type, rule_tag, arity, types, name)
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

/// Export of one FDiagram for Python materialisation:
/// (dom_atoms, [(name, dom_atoms, cod_atoms, kind, z, is_dagger, offset)], cod_atoms)
type BoxExport = (String, AtomList, AtomList, u8, i32, bool, u32);
type DiagramExport = (AtomList, Vec<BoxExport>, AtomList);

fn ty_export(ty: &FTy) -> AtomList {
    ty.0.iter().map(|&a| (atom_name(a), atom_z(a))).collect()
}

#[pyclass]
struct RsDiagram {
    inner: FDiagram,
}

#[pymethods]
impl RsDiagram {
    /// Materialisation data; (name, z) atoms re-interned Python-side.
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

/// Build a batch of diagrams from post-order build programs, in parallel.
#[pyfunction]
fn build_diagrams(programs: Vec<Vec<PyNode>>) -> Vec<RsDiagram> {
    programs
        .into_par_iter()
        .map(|prog| {
            let nodes: Vec<Node> = prog.into_iter().map(decode_node).collect();
            RsDiagram { inner: assemble(&nodes) }
        })
        .collect()
}
```

Register in the `#[pymodule]` fn:

```rust
    m.add_class::<RsDiagram>()?;
    m.add_function(wrap_pyfunction!(build_diagrams, m)?)?;
```

NOTE: `build_diagrams` releases the GIL implicitly only if it takes no
`Python` token and uses owned data — `Vec<Vec<PyNode>>` is fully decoded
into owned Rust before the parallel map, so rayon runs GIL-free. Verify
the decode happens before `into_par_iter` (it does: `programs` is already
an owned `Vec`). If PyO3 extraction of the nested tuple type fails to
compile, decode in a sequential pre-pass to `Vec<Vec<Node>>` first, then
`into_par_iter` over that.

- [ ] **Step 2: Python materializer** in
`lambeq/backend/fast/convert.py` — add:

```python
def rs_to_fast(rsdiagram) -> FDiagram:
    """Materialise an FDiagram from a bobcat_rs RsDiagram export.

    The Rust atom ids are re-interned here via atom(name, z), so the
    result carries Python atom ids.
    """
    from lambeq.backend.fast.diagram import FBox, FDiagram
    from lambeq.backend.fast.types import FTy, atom

    def _fty(atoms):
        return FTy(tuple(atom(name, z) for name, z in atoms))

    dom_atoms, terms, cod_atoms = rsdiagram.export()
    fterms = []
    for name, b_dom, b_cod, kind, z, is_dagger, off in terms:
        box = FBox(name, _fty(b_dom), _fty(b_cod), kind, z, bool(is_dagger))
        fterms.append((box, int(off)))
    return FDiagram(_fty(dom_atoms), tuple(fterms), _fty(cod_atoms))
```

(`FBox`/`FDiagram` accept these args per `lambeq/backend/fast/diagram.py`;
validation runs if enabled — keep it on in tests.)

- [ ] **Step 3: build-program emitter + backend switch** in
`lambeq/text2diagram/ccg_tree.py`. Add a module-level emitter that mirrors
`_to_fast_diagram`/`_fast_rule_layer` but appends program nodes in
post-order, and extend `to_fast_diagram` with a `backend` kwarg:

```python
# rule-tag table (must match rust/src/build.rs)
_RULE_TAGS = {
    CCGRule.FORWARD_APPLICATION: 0,
    CCGRule.BACKWARD_APPLICATION: 1,
    CCGRule.FORWARD_COMPOSITION: 2,
    CCGRule.BACKWARD_COMPOSITION: 3,
    CCGRule.FORWARD_CROSSED_COMPOSITION: 4,
    CCGRule.BACKWARD_CROSSED_COMPOSITION: 5,
    CCGRule.GENERALIZED_FORWARD_COMPOSITION: 6,
    CCGRule.GENERALIZED_BACKWARD_COMPOSITION: 7,
    CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION: 8,
    CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION: 9,
    CCGRule.FORWARD_TYPE_RAISING: 10,
    CCGRule.BACKWARD_TYPE_RAISING: 11,
    CCGRule.REMOVE_PUNCTUATION_LEFT: 12,
    CCGRule.REMOVE_PUNCTUATION_RIGHT: 13,
}
```

The emitter (place near `_fast_rule_layer`). `CCGRule` and `CCGType` are
already imported there; `_atoms(grammar_ty)` returns `[(ob.name, ob.z)
for ob in grammar_ty.objects]` (confirm the attribute that lists atoms +
their `.name`/`.z` against `lambeq/backend/grammar.py` `Ty`; it is what
`convert.ty_to_fast` iterates):

```python
def _atoms(gty):
    return [(ob.name, ob.z) for ob in gty]   # Ty iterates its atoms


def _emit_program(tree, out):
    """Append tree's post-order build-program nodes to `out`."""
    if tree.rule == CCGRule.LEXICAL:
        if tree.biclosed_type == CCGType.PUNCTUATION:
            out.append((1, 0, 0, [], ''))
        else:
            cod = _atoms(tree.biclosed_type.to_grammar())
            out.append((0, 0, 0, [cod], tree.text))
        return

    if tree.rule == CCGRule.UNARY:
        if tree.biclosed_type.is_over:
            left = _atoms(tree.biclosed_type.left.to_grammar())
            rg = tree.biclosed_type.right.to_grammar().l
            right = _atoms(rg)
        else:
            left = _atoms(tree.biclosed_type.left.to_grammar().r)
            right = _atoms(tree.biclosed_type.right.to_grammar())
        for child in tree.children:
            _emit_program(child, out)
        # UNARY_SWAP types = [right, left]
        out.append((2, 0, 1, [right, left], ''))
        return

    for child in tree.children:
        _emit_program(child, out)
    tag, args = _rule_program(tree.rule,
                              [c.biclosed_type for c in tree.children],
                              tree.biclosed_type)
    out.append((3, tag, len(tree.children), args, ''))


def _rule_program(rule, dom, cod):
    """(tag, [arg atom-lists]) mirroring _fast_rule_layer's arg extraction."""
    def a(gty):
        return _atoms(gty.to_grammar()) if hasattr(gty, 'to_grammar') \
            else _atoms(gty)

    tag = _RULE_TAGS[rule]
    if rule in (CCGRule.FORWARD_TYPE_RAISING, CCGRule.BACKWARD_TYPE_RAISING):
        return tag, [a(cod.result), a(dom[0])]
    left, right = dom
    if rule == CCGRule.FORWARD_APPLICATION:
        return tag, [a(left.result), a(right)]
    if rule == CCGRule.BACKWARD_APPLICATION:
        return tag, [a(left), a(right.result)]
    if rule == CCGRule.FORWARD_COMPOSITION:
        return tag, [a(left.left), a(left.right), a(right.right)]
    if rule == CCGRule.BACKWARD_COMPOSITION:
        return tag, [a(left.left), a(left.right), a(right.right)]
    if rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
        return tag, [a(left.left), a(left.right), a(right.left)]
    if rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
        return tag, [a(left.right), a(left.left), a(right.right)]
    if rule == CCGRule.GENERALIZED_FORWARD_COMPOSITION:
        mid = left.argument.to_grammar()
        return tag, [a(left.result), _atoms(mid),
                     _atoms(right.to_grammar()[len(mid):])]
    if rule == CCGRule.GENERALIZED_BACKWARD_COMPOSITION:
        mid = right.argument.to_grammar()
        lg = left.to_grammar()
        return tag, [_atoms(lg[:len(lg) - len(mid)]), _atoms(mid),
                     a(right.result)]
    if rule == CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION:
        mid = left.left.to_grammar()
        gl, join, gr = right.split(left.right)
        return tag, [_atoms(mid), _atoms(gl), _atoms(join), _atoms(gr)]
    if rule == CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION:
        mid = right.right.to_grammar()
        gl, join, gr = left.split(right.left)
        return tag, [_atoms(mid), _atoms(gl), _atoms(join), _atoms(gr)]
    if rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
        return tag, [a(right)]
    if rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
        return tag, [a(left)]
    raise AssertionError(f'unreachable rule {rule}')
```

Extend `to_fast_diagram`:

```python
    def to_fast_diagram(self, backend='python'):
        """Build a fast-core FDiagram. backend='python' (default) uses the
        in-process recursion; 'rust' offloads assembly to bobcat_rs."""
        if backend == 'rust':
            from lambeq.text2diagram.ccg_tree import trees_to_fast_diagrams
            return trees_to_fast_diagrams([self], backend='rust')[0]
        resolved = self.collapse_noun_phrases()._resolved()
        words, grammar = resolved._to_fast_diagram()
        return words >> grammar
```

Add the batch entry point (module level):

```python
def trees_to_fast_diagrams(trees, backend='python'):
    """Build FDiagrams for many trees. backend='rust' batches the
    assembly into one bobcat_rs.build_diagrams call (rayon-parallel)."""
    if backend != 'rust':
        return [t.to_fast_diagram(backend='python') for t in trees]
    import bobcat_rs
    from lambeq.backend.fast import convert
    programs = []
    for t in trees:
        prog = []
        _emit_program(t.collapse_noun_phrases()._resolved(), prog)
        programs.append(prog)
    rsdiagrams = bobcat_rs.build_diagrams(programs)
    return [convert.rs_to_fast(rs) for rs in rsdiagrams]
```

- [ ] **Step 4: build the extension**

`~/.pyenv/versions/qnlp/bin/maturin develop --release -m rust/Cargo.toml`
(600000ms). Expected: builds, installs `bobcat_rs` with the new symbols.
Confirm: `$PY -c "import bobcat_rs; print(hasattr(bobcat_rs,
'build_diagrams'), hasattr(bobcat_rs, 'RsDiagram'))"` → `True True`.

- [ ] **Step 5: corpus gates** (`tests/backend/test_rust_construct.py`):

```python
import pytest

pytest.importorskip('bobcat_rs')

from lambeq.backend.fast import convert
from lambeq.text2diagram.ccg_tree import trees_to_fast_diagrams


def test_rust_matches_to_diagram(bobcat_trees):
    assert bobcat_trees
    fasts = trees_to_fast_diagrams(bobcat_trees, backend='rust')
    for t, fd in zip(bobcat_trees, fasts):
        assert convert.to_grammar(fd) == t.to_diagram()


def test_rust_equals_python(bobcat_trees):
    fasts = trees_to_fast_diagrams(bobcat_trees, backend='rust')
    for t, fd in zip(bobcat_trees, fasts):
        assert fd == t.to_fast_diagram(backend='python')
```

Run: `$PY -m pytest tests/backend/test_rust_construct.py -q` (600000ms).
Both gates must pass (validation ON via conftest). Debug a mismatch by
diffing `convert.to_grammar(fd)` vs `t.to_diagram()` reprs and finding
the diverging rule; the bug is in the Rust port or the emitter — the
oracle is the truth, do not weaken the gate.

- [ ] **Step 6: flake8 + commit**

```bash
$PY -m flake8 lambeq/text2diagram/ccg_tree.py lambeq/backend/fast/convert.py tests/backend/test_rust_construct.py
git add rust/src/lib.rs lambeq/text2diagram/ccg_tree.py lambeq/backend/fast/convert.py tests/backend/test_rust_construct.py
git commit -m "Wire the Rust construction kernel into to_fast_diagram

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Benchmark (batched, single vs rayon) + RESULTS + push

**Files:** Create `benchmarks/rust_construct_bench.py`; modify
`benchmarks/RESULTS.md`.

- [ ] **Step 1: write `benchmarks/rust_construct_bench.py`** — parse the
corpus once (rust backend), then time, over all parsed trees:
- Python direct: `[t.to_fast_diagram(backend='python') for t in trees]`
- Rust batched: `trees_to_fast_diagrams(trees, backend='rust')`
- the program-emit time alone (Python) vs the Rust `build_diagrams` time
  alone (to show the emit/assembly split and the FFI share)

Report ms/diagram and diagrams/s for each, the speedup, and a comparison
to parser throughput. Structure like `benchmarks/fastdiag_bench.py`
(argparse, `--num`). To isolate rayon scaling, also time with
`RAYON_NUM_THREADS=1` vs unset (note: set via env before the run; print
which). Assert the Rust result equals the Python result over the corpus
before timing (so numbers are never from a broken build).

- [ ] **Step 2: run** `$PY benchmarks/rust_construct_bench.py
/tmp/coco_bench.txt --num 2000` (and a `RAYON_NUM_THREADS=1` run).
Capture ms/diagram + diagrams/s for python-direct, rust-1-thread,
rust-all-cores.

- [ ] **Step 3: append `## Rust construction (stage A)` to
`benchmarks/RESULTS.md`** — the measured table (python-direct vs rust
single vs rust rayon, ms/diagram + diagrams/s), the emit/assembly/FFI
split, throughput vs the ~1700 sent/s parser, and an honest verdict:
did Rust beat 0.76ms single-thread? does rayon throughput exceed the
parser (construction no longer the bottleneck)? If intern-table Mutex
contention caps rayon scaling, say so and note the pre-intern follow-up.

- [ ] **Step 4: full sweep + commit + push**

```bash
cargo test --manifest-path rust/Cargo.toml
$PY -m pytest tests/backend/test_rust_construct.py tests/backend/test_fast_construct.py tests/backend/test_fast_build.py -q
git add benchmarks/rust_construct_bench.py benchmarks/RESULTS.md
git commit -m "Benchmark the Rust construction kernel

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin rust-diagram-core
```

- [ ] **Step 5: report** — the differential result (corpus N, 0
mismatches), rust==python result, single-thread and rayon ms/diagram +
speedup vs Python-direct, whether construction now exceeds parser
throughput, and whether push succeeded.
