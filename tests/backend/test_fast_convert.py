import pytest

from lambeq.backend import grammar
from lambeq.backend.fast import FTy, convert


def test_ty_round_trip():
    n, s = grammar.Ty('n'), grammar.Ty('s')
    t = n @ s.l @ n.r
    assert convert.ty_to_grammar(convert.ty_to_fast(t)) == t


def test_ty_z_preserved():
    n = grammar.Ty('n')
    t = n.l.l
    fty = convert.ty_to_fast(t)
    from lambeq.backend.fast.types import atom_z
    assert atom_z(fty[0]) == -2
    assert convert.ty_to_grammar(fty) == t


def test_dim_round_trip():
    from lambeq.backend import tensor
    d = tensor.Dim(2, 4)
    assert convert.ty_to_grammar(convert.ty_to_fast(d)) == d


def test_simple_diagram_round_trip():
    n, s = grammar.Ty('n'), grammar.Ty('s')
    alice = grammar.Word('Alice', n)
    likes = grammar.Word('likes', n.r @ s @ n.l)
    bob = grammar.Word('Bob', n)
    d = grammar.Diagram.create_pregroup_diagram(
        words=[alice, likes, bob],
        morphisms=[(grammar.Cup, 3, 4), (grammar.Cup, 0, 1)])
    rt = convert.to_grammar(convert.to_fast(d))
    assert rt == d


def test_daggered_word_round_trip():
    n = grammar.Ty('n')
    box = grammar.Box('f', n, n)
    d = (grammar.Word('a', n).to_diagram() >> box >> box.dagger())
    rt = convert.to_grammar(convert.to_fast(d))
    assert rt == d


def test_empty_box_round_trip():
    box = grammar.Box('scalar', grammar.Ty(), grammar.Ty())
    d = box.to_diagram()
    rt = convert.to_grammar(convert.to_fast(d))
    assert rt == d


def test_cups_match_grammar():
    from lambeq.backend.fast.diagram import cups
    n = grammar.Ty('n') @ grammar.Ty('s')
    g = grammar.Diagram.cups(n, n.r)
    f = cups(convert.ty_to_fast(n), convert.ty_to_fast(n.r))
    assert convert.to_grammar(f) == g


def test_frame_raises():
    pytest.importorskip('lambeq')
    n = grammar.Ty('n')
    frame = grammar.Frame('fr', n, n,
                          components=[grammar.Word('w', n)])
    with pytest.raises(NotImplementedError):
        convert.to_fast(frame.to_diagram())


def test_corpus_round_trip(bobcat_diagrams):
    assert bobcat_diagrams
    for d in bobcat_diagrams:
        assert convert.to_grammar(convert.to_fast(d)) == d


def test_to_fast_diagram_equivalent(bobcat_trees):
    assert bobcat_trees
    for t in bobcat_trees:
        assert (convert.to_grammar(t.to_fast_diagram())
                == t.to_diagram())
