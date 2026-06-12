// Port of lambeq/bobcat/parser.py:69-200 — the parse chart and beam cells.
//
// The tie-breaking in `Cell::add` and the beam-trim placement are
// load-bearing for the identical-trees guarantee; they mirror the Python
// oracle line-for-line.

use std::rc::Rc;

use rustc_hash::FxHashMap;

use crate::grammar::CatKey;
use crate::tree::Node;

const NEG_INF: f64 = f64::NEG_INFINITY;

// ---------------------------------------------------------------------------
// Cell (parser.py:69-163)
// ---------------------------------------------------------------------------

pub struct Cell {
    pub beam_size: usize,
    pub trees: Vec<Rc<Node>>,
    pub trees_map: FxHashMap<CatKey, Rc<Node>>,
    pub min_score: f64,
}

impl Cell {
    pub fn new(beam_size: usize) -> Cell {
        Cell {
            beam_size,
            trees: Vec::new(),
            trees_map: FxHashMap::default(),
            min_score: NEG_INF,
        }
    }

    /// parser.py:84-98. Binary search over the DESCENDING list of scores.
    pub fn find(&self, score: f64) -> usize {
        let trees = &self.trees;
        let mut lo = 0;
        let mut hi = trees.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let cmp = trees[mid].score.get();
            if score == cmp {
                return mid;
            } else if score > cmp {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    /// parser.py:100-163.
    pub fn add(&mut self, mut to_add: Vec<Rc<Node>>) -> isize {
        // Stable descending sort by score — Rust's sort_by is stable, so equal
        // scores keep input order, matching Python's `sorted(key=-score)`.
        // NaN-safe: treat as equal (stable sort keeps input order), rather
        // than panicking; NaN can only arise from pathological inputs like
        // input_tag_score_weight = 0 with -inf log-probs
        to_add.sort_by(|a, b| {
            b.score
                .get()
                .partial_cmp(&a.score.get())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut added: isize = 0;
        let b = self.beam_size;
        for tree in to_add {
            let score = tree.score.get();
            if b >= 1
                && self.trees.len() >= b
                && score < self.trees.last().unwrap().score.get()
            {
                break;
            }

            let key = CatKey(tree.cat.clone());

            // Check whether there exists a tree with the same category.
            let old = self.trees_map.get(&key).cloned();
            let insert = match old {
                None => true,
                Some(old_tree) => {
                    let old_score = old_tree.score.get();
                    let insert = score > old_score;
                    if insert {
                        let old_index = self.find(old_score);
                        let mut deleted = false;
                        let mut i = old_index;
                        while i < self.trees.len() {
                            if Rc::ptr_eq(&self.trees[i], &old_tree) {
                                self.trees.remove(i);
                                deleted = true;
                                break;
                            } else if self.trees[i].score.get() != old_score {
                                break;
                            }
                            i += 1;
                        }
                        if !deleted {
                            for j in (0..old_index).rev() {
                                if Rc::ptr_eq(&self.trees[j], &old_tree) {
                                    self.trees.remove(j);
                                    break;
                                }
                            }
                        }
                    }
                    insert
                }
            };

            if insert {
                let idx = self.find(score);
                self.trees.insert(idx, tree.clone());
                self.trees_map.insert(key, tree);
                added += 1;

                // Beam-trim block (parser.py:154-162), guarded like the Python
                // try/except IndexError: only act when trees[b-1] exists, and
                // trim only when trees[b] also exists and is below the cutoff.
                if b >= 1 && b - 1 < self.trees.len() {
                    let cutoff = self.trees[b - 1].score.get();
                    self.min_score = cutoff;
                    if b < self.trees.len() && self.trees[b].score.get() < cutoff {
                        added -= (self.trees.len() - b) as isize;
                        for t in &self.trees[b..] {
                            self.trees_map.remove(&CatKey(t.cat.clone()));
                        }
                        self.trees.truncate(b);
                    }
                }
            }
        }
        added
    }
}

// ---------------------------------------------------------------------------
// Chart (parser.py:166-200)
// ---------------------------------------------------------------------------

pub struct Chart {
    pub beam_size: usize,
    pub cells: FxHashMap<(u32, u32), Cell>,
    pub parse_tree_count: isize,
}

impl Chart {
    pub fn new(beam_size: usize) -> Chart {
        Chart {
            beam_size,
            cells: FxHashMap::default(),
            parse_tree_count: 0,
        }
    }

    /// parser.py:183-188.
    pub fn min_score(&self, start: u32, end: u32) -> f64 {
        self.cells
            .get(&(start, end))
            .map(|c| c.min_score)
            .unwrap_or(NEG_INF)
    }

    /// The trees list for a span, or None when the cell is absent
    /// (Python's `chart[span]` raising KeyError).
    pub fn trees(&self, start: u32, end: u32) -> Option<&Vec<Rc<Node>>> {
        self.cells.get(&(start, end)).map(|c| &c.trees)
    }

    /// parser.py:190-200.
    pub fn add(&mut self, start: u32, end: u32, to_add: Vec<Rc<Node>>) {
        if to_add.is_empty() {
            return;
        }
        let beam = self.beam_size;
        let cell = self
            .cells
            .entry((start, end))
            .or_insert_with(|| Cell::new(beam));
        self.parse_tree_count += cell.add(to_add);
    }
}
