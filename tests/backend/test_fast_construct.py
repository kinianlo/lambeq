from lambeq.backend.fast import convert


def test_corpus_construction_matches_oracle(bobcat_trees):
    assert bobcat_trees
    for t in bobcat_trees:
        assert (convert.to_grammar(t.to_fast_diagram())
                == t.to_diagram())


def test_construction_no_grammar_diagram(bobcat_trees):
    from lambeq.backend.fast import FDiagram
    for t in bobcat_trees[:5]:
        assert isinstance(t.to_fast_diagram(), FDiagram)
