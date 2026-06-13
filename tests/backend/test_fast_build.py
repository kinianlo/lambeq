import pytest

from lambeq.backend import grammar
from lambeq.backend.fast import build, convert
from lambeq.backend.fast.diagram import CUP, SWAP


def _g(ty):
    return convert.ty_to_fast(ty)


N, S, P = grammar.Ty('n'), grammar.Ty('s'), grammar.Ty('p')


@pytest.mark.parametrize('gfn,bfn,args', [
    (grammar.Diagram.fa, build.fa, (N, S)),
    (grammar.Diagram.fa, build.fa, (N @ S, P)),
    (grammar.Diagram.ba, build.ba, (N, S)),
    (grammar.Diagram.ba, build.ba, (N, S @ P)),
    (grammar.Diagram.fc, build.fc, (N, S, P)),
    (grammar.Diagram.bc, build.bc, (N, S, P)),
    (grammar.Diagram.fx, build.fx, (N, S, P)),
    (grammar.Diagram.bx, build.bx, (N, S, P)),
])
def test_combinator_matches_grammar(gfn, bfn, args):
    expected = gfn(*args)
    got = convert.to_grammar(bfn(*[_g(a) for a in args]))
    assert got == expected, (gfn.__name__, args)


def test_swaps_matches_grammar():
    for left, right in [(N, S), (N @ S, P), (N @ S, P @ N), (N, S @ P)]:
        expected = grammar.Diagram.swap(left, right)
        # grammar.Swap.__new__ returns a Box for atomic inputs;
        # normalise to Diagram so the comparison is type-consistent.
        if not isinstance(expected, grammar.Diagram):
            expected = expected.to_diagram()
        got = convert.to_grammar(build.swaps(_g(left), _g(right)))
        assert got == expected, (left, right)


def test_type_raising_matches_apply():
    T, A = S, N
    got = convert.to_grammar(build.ftr(_g(T), _g(A)))
    assert got == grammar.Diagram.caps(T, T.l) @ grammar.Id(A)
    got_b = convert.to_grammar(build.btr(_g(T), _g(A)))
    assert got_b == grammar.Id(A) @ grammar.Diagram.caps(T.r, T)


def test_gfc_gbc_match_grammar():
    L, M, tail = N, S, P
    got = convert.to_grammar(build.gfc(_g(L), _g(M), _g(tail)))
    assert got == (grammar.Id(L) @ grammar.Diagram.cups(M.l, M)
                   @ grammar.Id(tail))
    pre = P
    got_b = convert.to_grammar(build.gbc(_g(pre), _g(M), _g(N)))
    assert got_b == (grammar.Id(pre) @ grammar.Diagram.cups(M, M.r)
                     @ grammar.Id(N))


def test_swaps_box_kinds():
    d = build.swaps(_g(N @ S), _g(P))
    assert all(b.kind == SWAP for b, _ in d.terms)
    assert d.dom == _g(N @ S @ P) and d.cod == _g(P @ N @ S)


def test_fa_is_cups():
    d = build.fa(_g(N), _g(S))
    assert all(b.kind == CUP for b, _ in d.terms)


def test_gfx_matches_grammar_oracle():
    mid, l, join, r = S, N, P, N
    got = convert.to_grammar(build.gfx(_g(mid), _g(l), _g(join), _g(r)))
    expected = (
        grammar.Diagram.swap(mid @ join.l, l) @ grammar.Id(join)
        >> grammar.Id(l @ mid) @ grammar.Diagram.cups(join.l, join)
    ) @ grammar.Id(r)
    if not isinstance(expected, grammar.Diagram):
        expected = expected.to_diagram()
    assert got == expected, ('gfx oracle mismatch', repr(got), repr(expected))


def test_gbx_matches_grammar_oracle():
    mid, l, join, r = S, N, P, N
    got = convert.to_grammar(build.gbx(_g(mid), _g(l), _g(join), _g(r)))
    expected = grammar.Id(l) @ (
        grammar.Id(join) @ grammar.Diagram.swap(r, join.r @ mid)
        >> grammar.Diagram.cups(join, join.r) @ grammar.Id(mid @ r)
    )
    if not isinstance(expected, grammar.Diagram):
        expected = expected.to_diagram()
    assert got == expected, ('gbx oracle mismatch', repr(got), repr(expected))


def test_type_raising_multi_atom():
    T = S @ N
    got = convert.to_grammar(build.ftr(_g(T), _g(N)))
    expected = grammar.Diagram.caps(T, T.l) @ grammar.Id(N)
    if not isinstance(expected, grammar.Diagram):
        expected = expected.to_diagram()
    assert got == expected
