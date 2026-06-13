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
from lambeq.backend import grammar
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
