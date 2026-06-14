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
