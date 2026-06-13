import pytest

from lambeq.backend.grammar import Box, Cap, Cup, Id, Ty

from lambeq.backend.fast import convert
from lambeq.backend.fast.normal import normal_form, remove_snakes


# ---------------------------------------------------------------------------
# Oracle differential on the corpus.
# ---------------------------------------------------------------------------
def test_remove_snakes_matches_oracle(bobcat_diagrams):
    for g in bobcat_diagrams:
        oracle = g.remove_snakes()
        fast = convert.to_grammar(remove_snakes(convert.to_fast(g)))
        assert fast == oracle, g


def test_normal_form_matches_oracle(bobcat_diagrams):
    for g in bobcat_diagrams[:15]:
        oracle = g.normal_form()
        fast = convert.to_grammar(normal_form(convert.to_fast(g)))
        assert fast == oracle, g


# ---------------------------------------------------------------------------
# Synthetic snakes -- the corpus is mostly snake-free, so build genuine
# snakes in BOTH representations and assert the fast port unsnakes them
# to the same diagram the oracle produces.
# ---------------------------------------------------------------------------
_n = Ty('n')


def _left_snake():
    # Id @ Cap >> Cup @ Id  on n : dom n, cod n, yanks to Id(n).
    cap = Cap(_n.r, _n)            # cod = n.r @ n
    cup = Cup(_n, _n.r)           # dom = n  @ n.r
    return Id(_n) @ cap >> cup @ Id(_n)


def _right_snake():
    # Cap @ Id >> Id @ Cup  on n : dom n, cod n, yanks to Id(n).
    cap = Cap(_n, _n.l)           # cod = n @ n.l
    cup = Cup(_n.l, _n)          # dom = n.l @ n
    return cap @ Id(_n) >> Id(_n) @ cup


def test_remove_snakes_left_snake():
    g = _left_snake()
    fd = convert.to_fast(g)
    assert len(fd.terms) == 2          # cap + cup before removal
    out = remove_snakes(fd)
    assert len(out.terms) == 0         # genuinely straightened
    assert convert.to_grammar(out) == g.remove_snakes() == Id(_n)


def test_remove_snakes_right_snake():
    g = _right_snake()
    fd = convert.to_fast(g)
    assert len(fd.terms) == 2
    out = remove_snakes(fd)
    assert len(out.terms) == 0
    assert convert.to_grammar(out) == g.remove_snakes() == Id(_n)


def test_normal_form_left_snake():
    g = _left_snake()
    fast = convert.to_grammar(normal_form(convert.to_fast(g)))
    assert fast == g.normal_form() == Id(_n)


def test_remove_snakes_no_op_on_word_with_width2_cod():
    # A word with a width-2 cod is cap-SHAPED but not a cap: removal is
    # a no-op (guards the kind+shape detector against false positives).
    from lambeq.backend.grammar import Word
    g = Word('w', _n @ _n.l).to_diagram()
    fd = convert.to_fast(g)
    out = remove_snakes(fd)
    assert out.terms == fd.terms
    assert convert.to_grammar(out) == g.remove_snakes() == g


# ---------------------------------------------------------------------------
# Obstruction regression tests -- exercises obstruction-sliding loops
# in _unsnake, which the corpus tests rarely hit (corpus is mostly
# snake-free and short).
# ---------------------------------------------------------------------------
_s = Ty('s')


def test_remove_snakes_left_snake_right_obstruction():
    # Left snake (Id @ Cap >> Cup @ Id on n) with a box to the RIGHT of
    # the n.r wire being followed; exercises the right_obstruction loop.
    #
    # Layout: Id(n) @ Cap(n.r,n) @ Id(s)  -- cap at offset 1
    #         Id(n@n.r@n) @ Box('ro',s,s)   -- right obstruction at 3
    #         Cup(n,n.r) @ Id(n@s)          -- cup at offset 0
    g = (Id(_n) @ Cap(_n.r, _n) @ Id(_s)
         >> Id(_n @ _n.r @ _n) @ Box('ro', _s, _s)
         >> Cup(_n, _n.r) @ Id(_n @ _s))
    oracle = g.remove_snakes()
    assert oracle != g, 'snake was not found -- diagram construction error'
    fast = convert.to_grammar(remove_snakes(convert.to_fast(g)))
    assert fast == oracle


def test_remove_snakes_right_snake_left_obstruction():
    # Right snake (Cap(n,n.l) @ Id(n) >> Id(n) @ Cup(n.l,n)) with a box
    # to the LEFT of the n.l wire being followed; exercises the
    # left_obstruction loop on the right-snake branch.
    #
    # Layout: Id(s) @ Cap(n,n.l) @ Id(n)  -- cap at offset 1
    #         Box('lo2',s,s) @ Id(n @ n.l @ n)  -- left obstruction at 0
    #         Id(s @ n) @ Cup(n.l,n)            -- cup at offset 2
    g = (Id(_s) @ Cap(_n, _n.l) @ Id(_n)
         >> Box('lo2', _s, _s) @ Id(_n @ _n.l @ _n)
         >> Id(_s @ _n) @ Cup(_n.l, _n))
    oracle = g.remove_snakes()
    assert oracle != g, 'snake was not found -- diagram construction error'
    fast = convert.to_grammar(remove_snakes(convert.to_fast(g)))
    assert fast == oracle


def test_remove_snakes_left_snake_both_obstructions():
    # Left snake with boxes on BOTH sides of the n.r wire; exercises
    # both obstruction loops in the same unsnake call.
    #
    # Layout: Id(n) @ Cap(n.r,n) @ Id(n)
    #         Box('lo3',n,n) @ Id(n.r@n@n)   -- left obstruction at 0
    #         Id(n@n.r@n) @ Box('ro3',n,n)   -- right obstruction at 3
    #         Cup(n,n.r) @ Id(n@n)           -- cup at offset 0
    g = (Id(_n) @ Cap(_n.r, _n) @ Id(_n)
         >> Box('lo3', _n, _n) @ Id(_n.r @ _n @ _n)
         >> Id(_n @ _n.r @ _n) @ Box('ro3', _n, _n)
         >> Cup(_n, _n.r) @ Id(_n @ _n))
    oracle = g.remove_snakes()
    assert oracle != g, 'snake was not found -- diagram construction error'
    fast = convert.to_grammar(remove_snakes(convert.to_fast(g)))
    assert fast == oracle


def test_remove_snakes_doctest_example():
    # Replicates the module-level doctest from snake_removal.py:
    #   diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h
    # This snake has obstructions (f† and f flank the n.r wire).
    n = _n
    s = _s
    cup = Cup(n, n.r)
    cap = Cap(n.r, n)
    f = Box('f', n, n)
    g_box = Box('g', s @ n, n)
    h = Box('h', n, n @ s)
    diagram = g_box @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h
    oracle = diagram.remove_snakes()
    assert oracle != diagram, 'no snake found -- construction error'
    fast = convert.to_grammar(remove_snakes(convert.to_fast(diagram)))
    assert fast == oracle


def test_normal_form_non_connected_raises():
    # Tensoring two scalar boxes gives a diagram that is not connected;
    # normalize loops forever (oracle detects revisits and raises).
    # Both oracle and fast port must raise NotImplementedError.
    a = Box('a', Ty(), Ty()).to_diagram()
    b = Box('b', Ty(), Ty()).to_diagram()
    g = a @ b
    with pytest.raises(NotImplementedError):
        g.normal_form()
    with pytest.raises(NotImplementedError):
        normal_form(convert.to_fast(g))
