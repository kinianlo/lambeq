use std::collections::HashMap;

/// Shared pointer to an immutable Category. `Arc` (not `Rc`) so the shared
/// grammar/parser tables are `Send + Sync` and can be read concurrently by
/// rayon workers during batch parsing (Task 6).
pub type CatRef = std::sync::Arc<Category>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ATOM_STRINGS: &[&str] = &[
    "", "N", "NP", "S", "PP", "conj", ",", ";", ":", ".", "LQU", "RQU", "LRB", "RRB",
];

const FEATURE_STRINGS: &[&str] = &[
    "", "X", "adj", "as", "asup", "b", "bem", "dcl", "em", "expl", "for", "frg", "intj",
    "inv", "nb", "ng", "num", "poss", "pss", "pt", "q", "qem", "thr", "to", "wq",
];

pub const VARIABLES: &str = "+_YZWVUTRQAB";

pub const FEATURE_NONE: u8 = 0;
pub const FEATURE_X: u8 = 1;

pub const ATOM_N: u8 = 1;
pub const ATOM_NP: u8 = 2;
pub const ATOM_S: u8 = 3;
pub const ATOM_CONJ: u8 = 5;
pub const ATOM_COMMA: u8 = 6;
pub const ATOM_SEMICOLON: u8 = 7;

/// Type alias for a variable translation map (old var id -> new var id).
/// `None` means the var is absent from the map (Python would KeyError).
pub type VarTrans = [Option<u8>; 32];

// ---------------------------------------------------------------------------
// Hash helper (splitmix64-based, deterministic)
// ---------------------------------------------------------------------------

#[inline]
fn splitmix(mut x: u64) -> u64 {
    x = x ^ (x >> 30);
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x = x ^ (x >> 27);
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[inline]
fn hash2(a: u64, b: u64) -> u64 {
    splitmix(a ^ splitmix(b))
}

#[inline]
fn hash3(a: u64, b: u64, c: u64) -> u64 {
    splitmix(a ^ splitmix(b ^ splitmix(c)))
}

// ---------------------------------------------------------------------------
// Category
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Category {
    pub atom: u8,
    pub feature: u8,
    pub var: u8,
    pub has_relation: bool,
    /// 0 = atomic; b'/' or b'\\' = complex
    pub dir: u8,
    pub result: Option<CatRef>,
    pub argument: Option<CatRef>,
    #[allow(dead_code)] // read by type-raising in Task 3
    pub type_raising_dep_var: u8,
    pub hash: u64,
    /// Bitset of variable indices that appear in this subtree (var 0 excluded)
    pub vars: u32,
}

impl Category {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    pub fn new_atomic(
        atom: u8,
        feature: u8,
        var: u8,
        has_relation: bool,
    ) -> CatRef {
        let effective_feature = if feature == FEATURE_X { FEATURE_NONE } else { feature };
        let hash = hash2(atom as u64, effective_feature as u64);
        let vars = if var != 0 {
            debug_assert!(var < 32, "variable id out of bitset range");
            1u32 << var
        } else {
            0
        };
        CatRef::new(Category {
            atom,
            feature,
            var,
            has_relation,
            dir: 0,
            result: None,
            argument: None,
            type_raising_dep_var: 0,
            hash,
            vars,
        })
    }

    pub fn slash(
        result: CatRef,
        dir: u8,
        argument: CatRef,
        var: u8,
        has_relation: bool,
        type_raising_dep_var: u8,
    ) -> CatRef {
        let hash = hash3(result.hash, argument.hash, dir as u64);
        let mut vars = if var != 0 {
            debug_assert!(var < 32, "variable id out of bitset range");
            1u32 << var
        } else {
            0
        };
        vars |= result.vars | argument.vars;
        CatRef::new(Category {
            atom: 0,
            feature: 0,
            var,
            has_relation,
            dir,
            result: Some(result),
            argument: Some(argument),
            type_raising_dep_var,
            hash,
            vars,
        })
    }

    #[inline]
    pub fn is_atomic(&self) -> bool {
        self.dir == 0
    }

    /// Whether this is a backward complex category (`X\Y`).
    #[inline]
    pub fn bwd(&self) -> bool {
        self.dir == b'\\'
    }

    /// Whether this is a forward complex category (`X/Y`).
    #[inline]
    pub fn fwd(&self) -> bool {
        self.dir == b'/'
    }

    /// Whether the atom is a punctuation atom (`atom >= Atom.COMMA`).
    #[inline]
    pub fn atom_is_punct(&self) -> bool {
        self.atom >= ATOM_COMMA
    }

    #[inline]
    pub fn result_ref(&self) -> Option<&CatRef> {
        self.result.as_ref()
    }

    #[inline]
    pub fn argument_ref(&self) -> Option<&CatRef> {
        self.argument.as_ref()
    }

    // -----------------------------------------------------------------------
    // Translate (lexicon.py:124-148)
    // -----------------------------------------------------------------------

    /// Relabel variables via `var_map` and resolve the `X` feature to
    /// `feature` (when it is not NONE). Returns a fresh category.
    ///
    /// Mirrors `Category.translate`. Every non-zero var encountered is
    /// guaranteed present in `var_map` (Python would raise `KeyError`
    /// otherwise); var 0 maps to 0.
    pub fn translate(&self, var_map: &VarTrans, feature: u8) -> CatRef {
        let new_var = if self.var == 0 {
            0
        } else {
            var_map[self.var as usize]
                .expect("translate: variable missing from var_map")
        };

        if self.is_atomic() {
            let new_feature = if self.feature == FEATURE_X && feature != FEATURE_NONE {
                feature
            } else {
                self.feature
            };
            Category::new_atomic(self.atom, new_feature, new_var, self.has_relation)
        } else {
            let result = self
                .result
                .as_ref()
                .unwrap()
                .translate(var_map, feature);
            let argument = self
                .argument
                .as_ref()
                .unwrap()
                .translate(var_map, feature);
            Category::slash(result, self.dir, argument, new_var, self.has_relation, 0)
        }
    }

    // -----------------------------------------------------------------------
    // Equality (lexicon.py:197-211)
    // -----------------------------------------------------------------------

    pub fn equals(&self, other: &Category) -> bool {
        if self.hash != other.hash {
            return false;
        }
        self._equals(other)
    }

    fn _equals(&self, other: &Category) -> bool {
        if self.is_atomic() {
            if !other.is_atomic() {
                return false;
            }
            self.atom == other.atom
                && (self.feature == other.feature
                    || (self.atom == ATOM_S
                        && is_free(self.feature)
                        && is_free(other.feature)))
        } else {
            if other.is_atomic() {
                return false;
            }
            self.dir == other.dir
                && self.result.as_ref().unwrap()._equals(other.result.as_ref().unwrap())
                && self.argument.as_ref().unwrap()._equals(other.argument.as_ref().unwrap())
        }
    }

    // -----------------------------------------------------------------------
    // Matches (lexicon.py:217-225)
    // -----------------------------------------------------------------------

    pub fn matches(&self, other: &Category) -> bool {
        if self.is_atomic() {
            if !other.is_atomic() {
                return false;
            }
            self.atom == other.atom
                && (self.feature == FEATURE_NONE || self.feature == other.feature)
        } else {
            if other.is_atomic() {
                return false;
            }
            self.dir == other.dir
                && self.result.as_ref().unwrap().matches(other.result.as_ref().unwrap())
                && self.argument.as_ref().unwrap().matches(other.argument.as_ref().unwrap())
        }
    }

    // -----------------------------------------------------------------------
    // String representation (lexicon.py:150-182)
    // -----------------------------------------------------------------------

    /// Returns (output_string, updated_slot_counter).
    /// full=false → plain __str__, full=true → __repr__
    pub fn str_helper(&self, full: bool, mut sc: u32) -> (String, u32) {
        let mut output: String;

        if self.is_atomic() {
            let atom_str = ATOM_STRINGS[self.atom as usize];
            output = atom_str.to_string();
            if self.feature != FEATURE_NONE {
                output.push('[');
                output.push_str(FEATURE_STRINGS[self.feature as usize]);
                output.push(']');
            }
        } else {
            let res = self.result.as_ref().unwrap();
            let arg = self.argument.as_ref().unwrap();

            let (mut rs, sc2) = res.str_helper(full, sc);
            sc = sc2;
            if res.dir != 0 && !rs.ends_with('}') && !rs.ends_with('>') {
                rs = format!("({rs})");
            }

            let (mut as_, sc3) = arg.str_helper(full, sc);
            sc = sc3;
            if arg.dir != 0 && !as_.ends_with('}') && !as_.ends_with('>') {
                as_ = format!("({as_})");
            }

            let dir_char = self.dir as char;
            output = format!("{rs}{dir_char}{as_}");
        }

        if full && (self.var != 0 || self.has_relation) {
            if !self.is_atomic() {
                output = format!("({output})");
            }
            if self.var != 0 {
                let var_char = VARIABLES.as_bytes()[self.var as usize] as char;
                output.push('{');
                output.push(var_char);
                output.push('}');
            }
            if self.has_relation {
                sc += 1;
                output.push('<');
                output.push_str(&sc.to_string());
                output.push('>');
            }
        }

        (output, sc)
    }

    pub fn to_plain_str(&self) -> String {
        self.str_helper(false, 0).0
    }

    pub fn to_repr_str(&self) -> String {
        self.str_helper(true, 0).0
    }
}

#[inline]
fn is_free(feature: u8) -> bool {
    feature <= 1 // NONE or X
}

// ---------------------------------------------------------------------------
// Atom / Feature lookup helpers
// ---------------------------------------------------------------------------

fn atom_from_str(s: &str) -> Option<u8> {
    ATOM_STRINGS.iter().position(|&a| a == s).map(|i| i as u8)
}

fn feature_from_str(s: &str) -> Option<u8> {
    FEATURE_STRINGS.iter().position(|&f| f == s).map(|i| i as u8)
}

fn var_id(ch: char) -> Option<u8> {
    VARIABLES.find(ch).map(|i| i as u8)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    bytes: &'a [u8],
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Parser { bytes: s.as_bytes() }
    }

    /// Parse one category starting at `pos`.
    /// Returns (category, new_pos, new_slots).
    fn parse_cat(
        &self,
        type_raising_dep_var: u8,
        pos: usize,
        slots: u32,
        in_result: bool,
    ) -> (CatRef, usize, u32) {
        if self.bytes[pos] == b'(' {
            self.parse_complex(type_raising_dep_var, pos, slots, in_result)
        } else {
            self.parse_atomic(pos, slots)
        }
    }

    fn parse_complex(
        &self,
        type_raising_dep_var: u8,
        pos: usize,
        slots: u32,
        in_result: bool,
    ) -> (CatRef, usize, u32) {
        // Opening '('
        let pos = pos + 1;

        let (left, pos, slots) = self.parse_cat(0, pos, slots, in_result);
        let dir = self.bytes[pos];
        let pos = pos + 1;

        let slots = if in_result { slots + 1 } else { slots };

        let (right, pos, slots) = self.parse_cat(0, pos, slots, false);

        assert_eq!(self.bytes[pos], b')', "expected ')' at pos {pos}");
        let pos = pos + 1;

        // Parse optional {var} and/or <slot>
        let (var, has_relation, pos) = self.parse_var_slot(pos);

        let cat = Category::slash(left, dir, right, var, has_relation, type_raising_dep_var);
        (cat, pos, slots)
    }

    fn parse_atomic(&self, pos: usize, slots: u32) -> (CatRef, usize, u32) {
        // Match atom: [A-Z]+ | conj | [,.;:]
        let atom_end = if self.bytes[pos] == b',' || self.bytes[pos] == b'.'
            || self.bytes[pos] == b';' || self.bytes[pos] == b':'
        {
            pos + 1
        } else {
            let mut e = pos;
            // Try 'conj' first (lowercase)
            if self.bytes[pos..].starts_with(b"conj") {
                e = pos + 4;
            } else {
                while e < self.bytes.len() && self.bytes[e].is_ascii_uppercase() {
                    e += 1;
                }
            }
            e
        };

        let atom_str = std::str::from_utf8(&self.bytes[pos..atom_end]).unwrap();
        let atom = atom_from_str(atom_str)
            .unwrap_or_else(|| panic!("unknown atom: {atom_str}"));
        let mut pos = atom_end;

        // Optional [feature]
        let feature = if pos < self.bytes.len() && self.bytes[pos] == b'[' {
            let end = self.bytes[pos..].iter().position(|&b| b == b']').unwrap() + pos;
            let feat_str = std::str::from_utf8(&self.bytes[pos + 1..end]).unwrap();
            let f = feature_from_str(feat_str)
                .unwrap_or_else(|| panic!("unknown feature: {feat_str}"));
            pos = end + 1;
            f
        } else {
            FEATURE_NONE
        };

        // Optional {var} and <slot>
        let (var, has_relation, pos) = self.parse_var_slot(pos);

        let cat = Category::new_atomic(atom, feature, var, has_relation);
        (cat, pos, slots)
    }

    /// Parse optional `{VAR*}` then optional `<N>`.
    /// Returns (var, has_relation, new_pos).
    fn parse_var_slot(&self, mut pos: usize) -> (u8, bool, usize) {
        let var = if pos < self.bytes.len() && self.bytes[pos] == b'{' {
            let end = self.bytes[pos..].iter().position(|&b| b == b'}').unwrap() + pos;
            // The variable is the first char after '{'
            let var_ch = self.bytes[pos + 1] as char;
            let v = var_id(var_ch).unwrap_or_else(|| panic!("unknown var char: {var_ch}"));
            pos = end + 1;
            v
        } else {
            0
        };

        let has_relation = if pos < self.bytes.len() && self.bytes[pos] == b'<' {
            let end = self.bytes[pos..].iter().position(|&b| b == b'>').unwrap() + pos;
            pos = end + 1;
            true
        } else {
            false
        };

        (var, has_relation, pos)
    }
}

// ---------------------------------------------------------------------------
// Public parse entry points
// ---------------------------------------------------------------------------

thread_local! {
    static CACHE: std::cell::RefCell<HashMap<(String, u8), CatRef>> =
        std::cell::RefCell::new(HashMap::new());
}

pub fn parse(string: &str, type_raising_dep_var_char: char) -> CatRef {
    debug_assert_eq!(ATOM_STRINGS[ATOM_S as usize], "S");
    let tr_var = var_id(type_raising_dep_var_char)
        .unwrap_or_else(|| panic!("unknown tr_var char: {type_raising_dep_var_char}"));

    let key = (string.to_string(), tr_var);

    if let Some(cat) = CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return cat;
    }

    let cat = parse_uncached(string, tr_var);

    CACHE.with(|c| c.borrow_mut().insert(key, cat.clone()));
    cat
}

fn parse_uncached(string: &str, tr_var: u8) -> CatRef {
    let p = Parser::new(string);
    let (cat, pos, _) = p.parse_cat(tr_var, 0, 0, true);
    if pos == string.len() {
        return cat;
    }
    // Retry with wrapping parens
    let wrapped = format!("({string})");
    let p2 = Parser::new(&wrapped);
    let (cat2, pos2, _) = p2.parse_cat(tr_var, 0, 0, true);
    assert_eq!(pos2, wrapped.len(), "parse did not consume full string: {string}");
    cat2
}
