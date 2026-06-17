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

from lambeq.backend import grammar
from lambeq.backend.fast import FRewriter, convert
from lambeq.core.types import AtomicType
from lambeq.rewrite import Rewriter

N = AtomicType.NOUN
S = AtomicType.SENTENCE

EXTENDED = ['determiner', 'connector', 'coordination', 'curry',
            'prepositional_phrase', 'auxiliary',
            'object_rel_pronoun', 'subject_rel_pronoun']


# ---------------------------------------------------------------------------
# Corpus differential against the legacy Rewriter (the oracle).
# ---------------------------------------------------------------------------
def test_default_rules_match_oracle(bobcat_diagrams):
    assert bobcat_diagrams
    rw, fr = Rewriter(), FRewriter()
    for d in bobcat_diagrams:
        assert convert.to_grammar(fr(convert.to_fast(d))) == rw(d), d


def test_extended_rules_match_oracle(bobcat_diagrams):
    rw, fr = Rewriter(EXTENDED), FRewriter(EXTENDED)
    for d in bobcat_diagrams:
        assert convert.to_grammar(fr(convert.to_fast(d))) == rw(d), d


# ---------------------------------------------------------------------------
# Coverage instrumentation: how many diagrams actually get rewritten?
# A passing corpus differential is VACUOUS if no rule ever fires, so we
# count and report the changed-diagram counts for both rule sets.
# ---------------------------------------------------------------------------
def _count_changed(rules, diagrams):
    fr = FRewriter(rules)
    return sum(1 for d in diagrams
               if convert.to_grammar(fr(convert.to_fast(d))) != d)


def test_report_corpus_coverage(bobcat_diagrams):
    total = len(bobcat_diagrams)
    n_default = _count_changed(None, bobcat_diagrams)
    n_extended = _count_changed(EXTENDED, bobcat_diagrams)
    print(f'\n[FRewriter coverage] corpus size: {total}')
    print(f'[FRewriter coverage] default rules changed: {n_default}')
    print(f'[FRewriter coverage] extended rules changed: {n_extended}')
    # Guard the corpus differentials against vacuity: if no rule fires,
    # to_grammar(fr(...)) == rw(...) holds trivially (both are identity)
    # and validates nothing.
    assert n_default > 0, (
        f'default rules changed 0/{total} diagrams — '
        'corpus differential is vacuous')
    assert n_extended > 0, (
        f'extended rules changed 0/{total} diagrams — '
        'corpus differential is vacuous')


# ---------------------------------------------------------------------------
# Hand-built per-rule cases: guarantee each supported rule genuinely
# fires and that FRewriter mirrors Rewriter on it, regardless of the
# corpus contents.
# ---------------------------------------------------------------------------
_PER_RULE_DIAGRAMS = {
    # determiner: cod N << N == N @ N.l, words a/an/the
    'determiner': grammar.Word('the', N @ N.l).to_diagram(),
    # connector: cod S << S == S @ S.l, words and/but/that/...
    'connector': grammar.Word('that', S @ S.l).to_diagram(),
    # coordination: word 'and', cod a.r @ a @ a.l (here a = N)
    'coordination': grammar.Word('and', N.r @ N @ N.l).to_diagram(),
    # curry: any box whose cod has an adjoint at either end
    'curry': grammar.Word('runs', N.r @ S).to_diagram(),
    # prepositional_phrase: cod (N >> S) >> (N >> S << N)
    'prepositional_phrase': grammar.Word(
        'in', (N >> S) >> (N >> S << N)).to_diagram(),
    # postadverb: cod (N >> S) >> (N >> S), matched by cod (no words)
    'postadverb': grammar.Word(
        'quickly', (N >> S) >> (N >> S)).to_diagram(),
    # preadverb: cod (N >> S) << (N >> S), matched by cod (no words)
    'preadverb': grammar.Word(
        'very', (N >> S) << (N >> S)).to_diagram(),
    # auxiliary: same cod as preadverb, restricted to auxiliary words
    'auxiliary': grammar.Word(
        'is', (N >> S) << (N >> S)).to_diagram(),
}


@pytest.mark.parametrize('name', list(_PER_RULE_DIAGRAMS))
def test_per_rule_fires_and_matches_oracle(name):
    d = _PER_RULE_DIAGRAMS[name]
    rw, fr = Rewriter([name]), FRewriter([name])
    # The rule must genuinely change the diagram, else it is vacuous.
    assert rw(d) != d, f'{name}: hand-built diagram does not fire the rule'
    assert convert.to_grammar(fr(convert.to_fast(d))) == rw(d), name


# ---------------------------------------------------------------------------
# Composability: an FRewriter output feeds the rest of the fast pipeline
# (remove_cups -> FSpiderAnsatz -> to_contraction) and contracts to the
# same tensor as the legacy Rewriter -> RemoveCups -> SpiderAnsatz ->
# PytorchModel path. Proves FRewriter slots into the fast pipeline.
# ---------------------------------------------------------------------------
def test_rewriter_then_pipeline_numeric(bobcat_diagrams):
    torch = pytest.importorskip('torch')
    from collections import Counter

    from lambeq import PytorchModel, RemoveCupsRewriter, SpiderAnsatz
    from lambeq.backend.fast import FSpiderAnsatz, contraction
    from lambeq.backend.fast.normal import remove_cups
    from lambeq.backend.tensor import Dim

    ob = {t: Dim(2) for t in AtomicType}
    rw, fr = Rewriter(), FRewriter()
    rc = RemoveCupsRewriter()

    # Legacy reference circuits; group by output shape and keep the
    # modal group so get_diagram_output stacks a single batch.
    g = [SpiderAnsatz(ob)(rc(rw(d))) for d in bobcat_diagrams]
    shapes = [tuple(c.cod.dim) for c in g]
    modal = Counter(shapes).most_common(1)[0][0]
    keep = [i for i, s in enumerate(shapes) if s == modal]
    assert len(keep) >= 4
    diags = [bobcat_diagrams[i] for i in keep]
    g = [g[i] for i in keep]

    model = PytorchModel.from_diagrams(g)
    torch.manual_seed(0)
    model.initialise_weights()
    expected = model.get_diagram_output(g)
    weights = dict(zip(model.symbols, model.weights))

    fans = FSpiderAnsatz(ob)
    for d, e in zip(diags, expected):
        fd = fans(remove_cups(fr(convert.to_fast(d))))
        got = contraction.evaluate(contraction.to_contraction(fd), weights)
        assert torch.allclose(got, e, atol=1e-5), d
