# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge to the optional bobcat_rs Rust chart-parser extension."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from lambeq.bobcat.lexicon import Category
from lambeq.bobcat.tree import IndexedWord, ParseTree, Rule, Variable

if TYPE_CHECKING:
    from lambeq.bobcat.grammar import Grammar
    from lambeq.bobcat.parser import Sentence


def nodes_to_tree(nodes: list[tuple[str, str, str | None, int, int]]
                  ) -> ParseTree:
    """Rebuild a ParseTree from a bobcat_rs flat node list.

    Nodes are in post-order (children first, leaves in sentence order,
    root last). The rebuilt tree carries no dependency state: it exists
    for `_build_ccgtree` (structure, rules, categories, leaf words) and
    for `metadata['original']`.
    """
    trees: list[ParseTree] = []
    index = 0
    for rule_name, cat_str, word, left, right in nodes:
        cat = Category.parse(cat_str)
        var_map = {}
        if word is not None:
            index += 1
            var_map = {cat.var: Variable(IndexedWord(word, index))}
        trees.append(ParseTree(Rule[rule_name],
                               cat,
                               trees[left] if left >= 0 else None,
                               trees[right] if right >= 0 else None,
                               [], [], var_map))
    return trees[-1]


class RustBackend:
    """Drop-in for ChartParser backed by bobcat_rs.

    Satisfies the pieces of the ChartParser interface that
    BobcatParser uses: __call__ (single sentence), parse_batch,
    set_root_cats.
    """

    def __init__(self,
                 grammar: Grammar,
                 cats: Iterable[str],
                 root_cats: Iterable[str] | None,
                 eisner_normal_form: bool,
                 max_parse_trees: int,
                 beam_size: int,
                 input_tag_score_weight: float,
                 missing_cat_score: float,
                 missing_span_score: float) -> None:
        import bobcat_rs
        self._parser = bobcat_rs.RustChartParser(
            grammar.categories,
            [tuple(r) for r in grammar.binary_rules],
            [tuple(r) for r in grammar.type_changing_rules],
            [tuple(r) for r in grammar.type_raising_rules],
            list(cats),
            list(root_cats) if root_cats is not None else None,
            eisner_normal_form,
            max_parse_trees,
            beam_size,
            input_tag_score_weight,
            missing_cat_score,
            missing_span_score)

    def set_root_cats(self, root_cats: Iterable[str] | None = None) -> None:
        self._parser.set_root_cats(
            list(root_cats) if root_cats is not None else None)

    @staticmethod
    def _to_input(sentence: Sentence) -> tuple:
        supertags = [[(st.category, st.probability) for st in sts]
                     for sts in sentence.input_supertags]
        return (sentence.words, supertags, sentence.span_scores)

    def parse_batch(self,
                    sentences: Iterable[Sentence],
                    num_threads: int = 0) -> list[ParseTree | None]:
        """Parse Sentence objects; returns a list of ParseTree | None."""
        results = self._parser.parse_batch(
            [self._to_input(s) for s in sentences], num_threads)
        return [nodes_to_tree(nodes) if nodes is not None else None
                for nodes in results]

    def __call__(self, sentence: Sentence) -> _SingleResult:
        return _SingleResult(self.parse_batch([sentence])[0])


class _SingleResult:
    """Minimal stand-in for ParseResult covering only what BobcatParser uses: truthiness and index 0."""

    def __init__(self, tree: ParseTree | None) -> None:
        self._tree = tree

    def __bool__(self) -> bool:
        return self._tree is not None

    def __getitem__(self, index: int) -> ParseTree:
        if self._tree is None or index != 0:
            raise IndexError(index)
        return self._tree
