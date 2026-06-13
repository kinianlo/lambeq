import pytest

from lambeq.backend import fast
from lambeq.backend.fast import FBox, FDiagram, FTy
from lambeq.backend.fast.diagram import (CAP, CUP, PLAIN, SPIDER, SWAP,
                                         WORD, cap, caps, cup, cups,
                                         spider, swap, word)


def _n():
    return FTy.of('n')


def test_box_identity_and_hash():
    f = FBox('f', _n(), _n() @ _n())
    g = FBox('f', _n(), _n() @ _n())
    assert f == g and hash(f) == hash(g)
    assert f != FBox('g', _n(), _n() @ _n())
    assert f.kind == PLAIN


def test_compose_and_tensor():
    n = _n()
    f = FBox('f', n, n).to_diagram()
    g = FBox('g', n, n).to_diagram()
    d = f >> g
    assert d.dom == n and d.cod == n
    assert [b.name for b, _ in d.terms] == ['f', 'g']
    t = f @ g
    assert t.dom == n @ n and t.cod == n @ n
    assert [o for _, o in t.terms] == [0, 1]


def test_then_validation():
    n, s = FTy.of('n'), FTy.of('s')
    f = FBox('f', n, n).to_diagram()
    h = FBox('h', s, s).to_diagram()
    with pytest.raises(ValueError):
        f >> h
    with fast.no_validation():
        d = f >> h          # trusted mode: no check, garbage in garbage out
        assert d.cod == s


def test_identity_unit_laws():
    n = _n()
    f = FBox('f', n, n).to_diagram()
    assert (FDiagram.id(n) >> f).terms == f.terms
    assert (f @ FDiagram.id(FTy())).terms == f.terms


def test_cups_shape():
    n2 = _n() @ _n()
    d = cups(n2, n2.r)
    assert d.dom == n2 @ n2.r and d.cod == FTy()
    assert all(b.kind == CUP for b, _ in d.terms)
    assert len(d.terms) == 2


def test_caps_shape():
    n2 = _n() @ _n()
    d = caps(n2, n2.r)
    assert d.dom == FTy() and d.cod == n2 @ n2.r
    assert all(b.kind == CAP for b, _ in d.terms)
    assert len(d.terms) == 2


def test_dagger_round_trip():
    n = _n()
    f = FBox('f', n, n @ n).to_diagram()
    dd = f.dagger().dagger()
    assert dd.dom == f.dom and dd.cod == f.cod
    assert dd == f


def test_diagram_hash_eq_stored():
    n = _n()
    a = FBox('f', n, n).to_diagram() >> FBox('g', n, n).to_diagram()
    b = FBox('f', n, n).to_diagram() >> FBox('g', n, n).to_diagram()
    assert a == b and hash(a) == hash(b)
