from lambeq.backend.grammar import Cap, Cup, Id, Ty

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
