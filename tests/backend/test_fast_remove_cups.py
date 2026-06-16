"""Differential tests for the fast-core ``remove_cups``.

Gate shipped: BYTE-IDENTICAL.  ``to_grammar(remove_cups(to_fast(d)))``
is asserted equal to ``RemoveCupsRewriter()(d)`` over the whole Bobcat
corpus and two hand-built pregroup cases.
"""

from lambeq import RemoveCupsRewriter
from lambeq.backend.grammar import Cup, Diagram, Ty, Word
from lambeq.backend.fast import convert
from lambeq.backend.fast.normal import remove_cups


N = Ty('n')
S = Ty('s')


def test_remove_cups_matches_oracle(bobcat_diagrams):
    assert bobcat_diagrams
    rc = RemoveCupsRewriter()
    for d in bobcat_diagrams:
        oracle = rc(d)
        fast = convert.to_grammar(remove_cups(convert.to_fast(d)))
        assert fast == oracle, d


def test_remove_cups_reduces_cups(bobcat_diagrams):
    for d in bobcat_diagrams[:15]:
        before = sum(isinstance(b, Cup) for b in d.boxes)
        after = convert.to_grammar(remove_cups(convert.to_fast(d)))
        assert sum(isinstance(b, Cup) for b in after.boxes) <= before


def test_remove_cups_single_cup():
    # Alice likes Bob : (n) (n.r s n.l) (n)
    words = [Word('Alice', N), Word('likes', N.r @ S @ N.l), Word('Bob', N)]
    morphs = [(Cup, 0, 1), (Cup, 3, 4)]
    d = Diagram.create_pregroup_diagram(words, morphs)
    rc = RemoveCupsRewriter()
    fast = convert.to_grammar(remove_cups(convert.to_fast(d)))
    assert fast == rc(d)


def test_remove_cups_nested():
    # Two words joined by a *composite* (nested) pair of cups, which is
    # what exercises _compress_cups merging into a CUP_TOKEN box.
    w = Word('w', N @ N)
    e = Word('e', (N @ N).r)
    d = (w @ e) >> Diagram.cups(N @ N, (N @ N).r)
    rc = RemoveCupsRewriter()
    fast = convert.to_grammar(remove_cups(convert.to_fast(d)))
    assert fast == rc(d)
