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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
import pytest

from lambeq.backend import fast, grammar
from lambeq.backend.fast import FBox, FFunctor, FTy, convert
from lambeq.backend.fast.types import atom


def test_identity_functor_is_noop(bobcat_diagrams):
    ident = FFunctor(ob=lambda f, a: FTy((a,)),
                     ar=lambda f, b: b)
    for d in (convert.to_fast(g) for g in bobcat_diagrams[:10]):
        assert ident(d) == d


def test_functor_matches_grammar_oracle(bobcat_diagrams):
    """Wire-doubling functor: n -> n@n, s -> s, words double cod."""
    n = grammar.Ty('n')

    def g_ob(_, ty):
        return ty @ ty if ty == n else ty

    def g_ar(functor, box):
        # Words must map to Words (and plain boxes to plain boxes) so the
        # grammar-class identity matches what ``convert.to_grammar``
        # reconstructs from the kind-tagged FBox image.
        if isinstance(box, grammar.Word):
            return grammar.Word(box.name, functor(box.cod), box.z)
        return grammar.Box(box.name, functor(box.dom), functor(box.cod),
                           box.z)

    g_functor = grammar.Functor(grammar.grammar, ob=g_ob, ar=g_ar)

    n_id = atom('n')

    def f_ob(_, a):
        return FTy((a, a)) if a == n_id else FTy((a,))

    def f_ar(functor, box):
        return FBox(box.name, functor(box.dom), functor(box.cod),
                    box.kind, box.z, box.is_dagger, box.payload)

    f_functor = FFunctor(ob=f_ob, ar=f_ar)
    for g in bobcat_diagrams[:10]:
        expected = g_functor(g)
        got = convert.to_grammar(f_functor(convert.to_fast(g)))
        assert got == expected, g


def test_functor_matches_oracle_on_rotated_plain_box():
    """A z!=0 plain box must be unwound, mapped, and rotated back -
    exactly mirroring grammar.Box.apply_functor."""
    n = grammar.Ty('n')

    def g_ob(_, ty):
        return ty @ ty if ty == n else ty

    def g_ar(functor, box):
        return grammar.Box(box.name, functor(box.dom), functor(box.cod),
                           box.z)

    g_functor = grammar.Functor(grammar.grammar, ob=g_ob, ar=g_ar)

    g = (grammar.Box('f', n, n).rotate(1)).to_diagram()
    expected = g_functor(g)

    n_id = atom('n')

    def f_ob(_, a):
        return FTy((a, a)) if a == n_id else FTy((a,))

    def f_ar(functor, box):
        return FBox(box.name, functor(box.dom), functor(box.cod),
                    box.kind, box.z, box.is_dagger, box.payload)

    f_functor = FFunctor(ob=f_ob, ar=f_ar)
    got = convert.to_grammar(f_functor(convert.to_fast(g)))
    assert got == expected


def test_dict_ob_matches_callable_ob():
    """Dict-form FFunctor must produce the same result as the equivalent
    callable-form functor on a diagram with both mapped and unmapped wires."""
    from lambeq.backend.fast import convert

    n_id = atom('n')
    n_ty = FTy.of('n')
    s_ty = FTy.of('s')
    mapped_n = n_ty @ n_ty   # n -> n @ n

    # Build a small diagram: a Word with cod n @ s  (n is mapped, s is not)
    g = grammar.Word('the', grammar.Ty('n') @ grammar.Ty('s')).to_diagram()
    fd = convert.to_fast(g)

    def f_ob_callable(_, a):
        return mapped_n if a == n_id else FTy((a,))

    def f_ar(functor, box):
        return FBox(box.name, functor(box.dom), functor(box.cod),
                    box.kind, box.z, box.is_dagger, box.payload)

    callable_functor = FFunctor(ob=f_ob_callable, ar=f_ar)
    dict_functor = FFunctor(ob={n_id: mapped_n}, ar=f_ar)

    assert dict_functor(fd) == callable_functor(fd)


def test_functor_cache_returns_shared_object():
    calls = []

    def ar(functor, box):
        calls.append(box.name)
        return FBox(box.name.upper(), box.dom, box.cod)

    f = FFunctor(ob=lambda _, a: FTy((a,)), ar=ar)
    b = FBox('f', FTy.of('n'), FTy.of('n'))
    d = b.to_diagram() >> b.to_diagram()
    out = f(d)
    assert calls == ['f']                       # second occurrence cached
    assert out.terms[0][0] is out.terms[1][0]   # SHARED object, no copy


def test_validation_raises_on_wrong_image_shape():
    """Under fast.validation(), map_box raises TypeError when the image
    has mismatched dom/cod; under no_validation(), the same functor
    applies without error."""
    n = grammar.Ty('n')
    n_ty = FTy.of('n')

    # ar widens cod: maps n->n to n->n@n (wrong shape)
    def f_ar(functor, box):
        return FBox(box.name, n_ty, n_ty @ n_ty)

    f_functor = FFunctor(ob=lambda _, a: FTy((a,)), ar=f_ar)
    fd = convert.to_fast(grammar.Box('f', n, n).to_diagram())

    # Validation is on (autouse fixture), but make intent explicit.
    # Wrong image dom/cod must raise TypeError.
    with fast.validation():
        with pytest.raises(TypeError):
            f_functor(fd)

    # Image was not cached (error prevents caching); no_validation
    # must apply the same functor without raising.
    with fast.no_validation():
        f_functor(fd)  # must not raise


def test_functor_multi_term_rotated_image_vs_oracle():
    """ar maps a PLAIN box to a two-box diagram; applied to a z-rotated
    plain box, the fast result must match the grammar oracle."""
    n = grammar.Ty('n')

    # Grammar oracle: maps box 'f' to g >> h (a two-box diagram)
    g = grammar.Box('g', n, n)
    h = grammar.Box('h', n, n)

    def g_ar(functor, box):
        return g.to_diagram() >> h.to_diagram()

    g_functor = grammar.Functor(grammar.grammar, ob=lambda _, ty: ty,
                                ar=g_ar)

    # Grammar diagram: single z=1 rotated box
    gd = grammar.Box('f', n, n).rotate(1).to_diagram()
    expected = g_functor(gd)

    # Fast functor: ar returns an FDiagram with two plain boxes
    fg = FBox('g', FTy.of('n'), FTy.of('n'))
    fh = FBox('h', FTy.of('n'), FTy.of('n'))

    def f_ar(functor, box):
        return fg.to_diagram() >> fh.to_diagram()

    f_functor = FFunctor(ob=lambda _, a: FTy((a,)), ar=f_ar)
    fast_d = convert.to_fast(gd)
    got = convert.to_grammar(f_functor(fast_d))

    assert got == expected
