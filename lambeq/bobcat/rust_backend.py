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

from lambeq.bobcat.lexicon import Category
from lambeq.bobcat.tree import IndexedWord, ParseTree, Rule, Variable


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
