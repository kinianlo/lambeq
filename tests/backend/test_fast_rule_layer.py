"""Dispatch test of ``_fast_rule_layer`` vs ``CCGRule.apply`` oracle.

For each CCG rule the test builds a minimal (dom, cod) pair, calls
both paths, and asserts equality after converting the fast result back
to a ``grammar.Diagram``.  A mismatch is a real wiring bug in the
fast path; the oracle side is never edited to force green.

Rules covered (14 total):
  FA, BA, FC, BC, FX, BX, GFC, GBC, GFX, GBX, FTR, BTR, LP, RP.
"""
from lambeq.text2diagram.ccg_tree import _fast_rule_layer
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.backend.fast import convert


n = CCGType.NOUN
s = CCGType.SENTENCE
p = CCGType.PREPOSITIONAL_PHRASE
conj = CCGType.CONJUNCTION
punc = CCGType.PUNCTUATION


def _check(rule, dom, cod):
    """Assert fast path == oracle for a given (rule, dom, cod)."""
    expected = rule.apply(dom, cod)
    got = convert.to_grammar(_fast_rule_layer(rule, dom, cod))
    assert got == expected, (
        f'Rule {rule}: mismatch\n  expected: {expected}\n  got:      {got}'
    )


def test_forward_application():
    # X/Y + Y -> X  (s/n + n -> s)
    _check(CCGRule.FORWARD_APPLICATION, [s << n, n], s)


def test_backward_application():
    # Y + X\Y -> X  (n + s\n -> s)  [n>>s encodes s\n]
    _check(CCGRule.BACKWARD_APPLICATION, [n, n >> s], s)


def test_forward_composition():
    # X/Y + Y/Z -> X/Z  (s/n + n/p -> s/p)
    _check(CCGRule.FORWARD_COMPOSITION, [s << n, n << p], s << p)


def test_backward_composition():
    # Z\Y + X\Y -> X\Z  (n\p + s\n -> s\p)  [p>>n encodes n\p]
    _check(CCGRule.BACKWARD_COMPOSITION, [p >> n, n >> s], p >> s)


def test_forward_crossed_composition():
    # X/Y + Y\Z -> X\Z  (s/n + n\p -> s\p)
    _check(CCGRule.FORWARD_CROSSED_COMPOSITION, [s << n, p >> n], p >> s)


def test_backward_crossed_composition():
    # Y/Z + X\Y -> X/Z  (n/p + s\n -> s/p)
    _check(CCGRule.BACKWARD_CROSSED_COMPOSITION, [n << p, n >> s], s << p)


def test_generalized_forward_composition():
    # X/Y + (Y/Z)/W -> (X/Z)/W  (s/n + (n/p)/conj -> (s/p)/conj)
    _check(
        CCGRule.GENERALIZED_FORWARD_COMPOSITION,
        [s << n, (n << p) << conj],
        (s << p) << conj,
    )


def test_generalized_backward_composition():
    # (Z\Y)\W + X\Y -> (X\Z)\W  ((n\p)\conj + s\n -> (s\p)\conj)
    _check(
        CCGRule.GENERALIZED_BACKWARD_COMPOSITION,
        [conj >> (p >> n), n >> s],
        conj >> (p >> s),
    )


def test_generalized_forward_crossed_composition():
    # X/Y + (Y\Z)|... -> (X\Z)|...
    # s/n + ((n\p)/conj)\punc -> ((s\p)/conj)\punc
    _check(
        CCGRule.GENERALIZED_FORWARD_CROSSED_COMPOSITION,
        [s << n, punc >> ((p >> n) << conj)],
        punc >> ((p >> s) << conj),
    )


def test_generalized_backward_crossed_composition():
    # (Y/Z)|... + X\Y -> (X/Z)|...
    # ((n/p)\conj)/punc + s\n -> ((s/p)\conj)/punc
    _check(
        CCGRule.GENERALIZED_BACKWARD_CROSSED_COMPOSITION,
        [(conj >> (n << p)) << punc, n >> s],
        (conj >> (s << p)) << punc,
    )


def test_forward_type_raising():
    # A -> T/(T\A);  n -> s/(s\n)
    cod = s << (n >> s)
    _check(CCGRule.FORWARD_TYPE_RAISING, [n], cod)


def test_backward_type_raising():
    # A -> T\(T/A);  n -> s\(s/n)
    cod = (s << n) >> s
    _check(CCGRule.BACKWARD_TYPE_RAISING, [n], cod)


def test_remove_punctuation_left():
    # punc + X -> X
    _check(CCGRule.REMOVE_PUNCTUATION_LEFT, [punc, n], n)


def test_remove_punctuation_right():
    # X + punc -> X
    _check(CCGRule.REMOVE_PUNCTUATION_RIGHT, [n, punc], n)
