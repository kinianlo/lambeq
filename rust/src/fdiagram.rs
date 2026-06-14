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
