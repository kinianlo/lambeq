from lambeq.backend import fast
from lambeq.backend.fast.types import FTy, atom, atom_name, atom_z


def test_atom_interning():
    a = atom('n')
    assert atom('n') == a            # same id for same (name, z)
    assert atom('n', z=1) != a
    assert atom_name(a) == 'n' and atom_z(a) == 0


def test_fty_algebra():
    n, s = FTy.of('n'), FTy.of('s')
    ns = n @ s
    assert len(ns) == 2
    assert ns == FTy.of('n') @ FTy.of('s')
    assert hash(ns) == hash(FTy.of('n') @ FTy.of('s'))
    assert ns @ FTy() == ns          # empty type is the unit
    assert list(iter(ns)) == [n.atoms[0], s.atoms[0]]
    assert ns[0:1] == n


def test_adjoints_round_trip():
    n = FTy.of('n')
    assert n.l.r == n and n.r.l == n
    ns = FTy.of('n') @ FTy.of('s')
    assert ns.l.atoms == (FTy.of('s').l.atoms[0], FTy.of('n').l.atoms[0])
    assert ns.l.r == ns
    assert atom_z(n.l.atoms[0]) == atom_z(n.atoms[0]) - 1  # convention pinned


def test_validation_switch():
    assert fast.validation_enabled()
    fast.set_validation(False)
    assert not fast.validation_enabled()
    fast.set_validation(True)
    with fast.no_validation():
        assert not fast.validation_enabled()
    assert fast.validation_enabled()
