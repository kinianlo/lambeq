import itertools
import json
from pathlib import Path

import pytest

bobcat_rs = pytest.importorskip('bobcat_rs')

from lambeq.bobcat.lexicon import Category

MODEL_DIR = Path.home() / '.cache/lambeq/bobcat/bobcat'


@pytest.fixture(scope='module')
def grammar_data():
    with open(MODEL_DIR / 'grammar.json') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def all_category_strings(grammar_data):
    cats = set(grammar_data['categories'].keys())
    cats |= set(grammar_data['categories'].values())
    for left, right in grammar_data['binary_rules']:
        cats |= {left, right}
    for _, left, right, res, _ in grammar_data['type_changing_rules']:
        cats |= {left, res} | ({right} if right else set())
    for left, right, _ in grammar_data['type_raising_rules']:
        cats |= {left, right}
    return sorted(cats)


def test_category_parse_matches_python(all_category_strings):
    for s in all_category_strings:
        py = Category.parse(s)
        rs_str, rs_repr = bobcat_rs.debug_parse_category(s, '+')
        assert rs_str == str(py), s
        assert rs_repr == repr(py), s


def test_type_raising_var_parse(grammar_data):
    for _, tr_cat, var in grammar_data['type_raising_rules']:
        py = Category.parse(tr_cat, var)
        rs_str, rs_repr = bobcat_rs.debug_parse_category(tr_cat, var)
        assert (rs_str, rs_repr) == (str(py), repr(py)), (tr_cat, var)


def test_category_eq_and_matches(all_category_strings):
    sample = all_category_strings[::7][:60]
    for a, b in itertools.product(sample, repeat=2):
        pa, pb = Category.parse(a), Category.parse(b)
        assert bobcat_rs.debug_cat_eq(a, b) == (pa == pb), (a, b)
        assert bobcat_rs.debug_cat_matches(a, b) == pa.matches(pb), (a, b)


def test_category_roundtrip(all_category_strings):
    for s in all_category_strings:
        _, rs_repr = bobcat_rs.debug_parse_category(s, '+')
        rt_str, rt_repr = bobcat_rs.debug_parse_category(rs_repr, '+')
        assert rt_repr == rs_repr, s


def test_category_vars_bitset(all_category_strings):
    for s in all_category_strings:
        py_vars = sorted(Category.parse(s).vars)
        assert bobcat_rs.debug_cat_vars(s) == py_vars, s


@pytest.fixture(scope='module')
def py_rules_no_tc(grammar_data):
    from lambeq.bobcat.grammar import Grammar
    from lambeq.bobcat.rules import Rules
    grammar = Grammar(categories=grammar_data['categories'],
                      binary_rules=grammar_data['binary_rules'],
                      type_changing_rules=[],
                      type_raising_rules=[])
    marked_up = {plain: Category.parse(marked)
                 for plain, marked in grammar.categories.items()}
    return Rules(True, grammar, marked_up), marked_up


@pytest.fixture(scope='module')
def rs_rules_no_tc(grammar_data):
    return bobcat_rs.RustRules(grammar_data['categories'],
                               [tuple(r) for r in grammar_data['binary_rules']],
                               [], [], True)


@pytest.fixture(scope='module')
def py_rules_full(grammar_data):
    from lambeq.bobcat.grammar import Grammar
    from lambeq.bobcat.rules import Rules
    grammar = Grammar(**grammar_data)
    marked_up = {plain: Category.parse(marked)
                 for plain, marked in grammar.categories.items()}
    return Rules(True, grammar, marked_up), marked_up


@pytest.fixture(scope='module')
def rs_rules_full(grammar_data):
    return bobcat_rs.RustRules(
        grammar_data['categories'],
        [tuple(r) for r in grammar_data['binary_rules']],
        [tuple(r) for r in grammar_data['type_changing_rules']],
        [tuple(r) for r in grammar_data['type_raising_rules']],
        True)


def test_type_change_and_raise_match_python(py_rules_full, rs_rules_full):
    from lambeq.bobcat.tree import Lexical
    rules, marked_up = py_rules_full
    for cat_str, cat in marked_up.items():
        tree = Lexical(cat, 'w', 1)
        expected_tc = [(t.rule.name, repr(t.cat))
                       for t in rules.type_change([tree])]
        assert rs_rules_full.debug_type_change(cat_str) == expected_tc, cat_str
        expected_tr = [(t.rule.name, repr(t.cat))
                       for t in rules.type_raise([tree])]
        assert rs_rules_full.debug_type_raise(cat_str) == expected_tr, cat_str


def test_full_combine_repr_matches_python(grammar_data, py_rules_full,
                                          rs_rules_full):
    from lambeq.bobcat.tree import Lexical
    rules, marked_up = py_rules_full
    checked = 0
    for left_str, right_str in grammar_data['binary_rules']:
        left_key = str(Category.parse(left_str))
        right_key = str(Category.parse(right_str))
        if left_key not in marked_up or right_key not in marked_up:
            continue
        left = Lexical(marked_up[left_key], 'l', 1)
        right = Lexical(marked_up[right_key], 'r', 2)
        expected = [(t.rule.name, repr(t.cat))
                    for t in rules.combine(left, right)]
        got = rs_rules_full.debug_combine_repr(left_key, right_key)
        assert got == expected, (left_str, right_str)
        checked += 1
    assert checked > 2000


def test_combine_matches_python_on_all_rule_instances(
        grammar_data, py_rules_no_tc, rs_rules_no_tc):
    from lambeq.bobcat.tree import Lexical
    rules, marked_up = py_rules_no_tc
    checked = 0
    for left_str, right_str in grammar_data['binary_rules']:
        # binary_rules strings are outer-parenthesised (e.g. '(NP/PP)') while
        # the categories table keys are canonical plain (e.g. 'NP/PP'); use the
        # canonical plain form so a Lexical tree can be built for each side.
        left_key = str(Category.parse(left_str))
        right_key = str(Category.parse(right_str))
        if left_key not in marked_up or right_key not in marked_up:
            continue
        left = Lexical(marked_up[left_key], 'l', 1)
        right = Lexical(marked_up[right_key], 'r', 2)
        expected = [(t.rule.name, str(t.cat))
                    for t in rules.combine(left, right)]
        got = rs_rules_no_tc.debug_combine(left_key, right_key)
        assert got == expected, (left_str, right_str)
        checked += 1
    assert checked > 1000


def test_left_comma_type_change_fires(py_rules_full, rs_rules_full):
    # pin the punct type-change gating: a known comma rule must produce
    # output through the full combine path, not vanish silently
    from lambeq.bobcat.tree import Lexical
    rules, marked_up = py_rules_full
    fired = []
    for left_str in (',', ';'):
        if left_str not in marked_up:
            continue
        for right_str in list(marked_up)[::3]:
            left = Lexical(marked_up[left_str], 'l', 1)
            right = Lexical(marked_up[right_str], 'r', 2)
            expected = [(t.rule.name, repr(t.cat))
                        for t in rules.combine(left, right)]
            got = rs_rules_full.debug_combine_repr(left_str, right_str)
            assert got == expected, (left_str, right_str)
            fired += [r for r, _ in expected if r in ('LP', 'RP')]
    assert fired


@pytest.fixture(scope='module')
def bobcat_parser():
    from lambeq import BobcatParser, VerbosityLevel
    return BobcatParser(verbose=VerbosityLevel.SUPPRESS.value)


@pytest.fixture(scope='module')
def rs_parser(grammar_data, bobcat_parser):
    return bobcat_rs.RustChartParser(
        grammar_data['categories'],
        [tuple(r) for r in grammar_data['binary_rules']],
        [tuple(r) for r in grammar_data['type_changing_rules']],
        [tuple(r) for r in grammar_data['type_raising_rules']],
        bobcat_parser.tagger.model.config.cats,
        None, True, 50000, 32, 1.0, 0.01, 1e-05)


SENTENCES = [
    'Alice likes Bob',
    'What Alice is and is not .',
    'I do not like Bob',
    'the old man sees a book about science in the park',
    'Alice likes Bob and the cat follows the dog while my neighbour '
    'reads a long and boring report before breakfast',
]


def _tagged(bobcat_parser, sentences):
    return bobcat_parser.tagger([s.split() for s in sentences],
                                verbose='suppress')


def _rust_inputs(bobcat_parser, tag_output):
    inputs = []
    for sent in tag_output.sentences:
        si = bobcat_parser._prepare_sentence(sent, tag_output.tags)
        supertags = [[(st.category, st.probability) for st in sts]
                     for sts in si.input_supertags]
        inputs.append((si.words, supertags, si.span_scores))
    return inputs


def test_serial_parse_equivalence(bobcat_parser, rs_parser):
    from lambeq.bobcat.rust_backend import nodes_to_tree
    from lambeq.text2diagram.model_based_reader.bobcat_parser import (
        BobcatParser)
    out = _tagged(bobcat_parser, SENTENCES)
    results = rs_parser.parse_batch(_rust_inputs(bobcat_parser, out))
    for sent, nodes in zip(out.sentences, results):
        si = bobcat_parser._prepare_sentence(sent, out.tags)
        py_tree = bobcat_parser.parser(si)[0]
        py_ccg = BobcatParser._build_ccgtree(py_tree)
        assert nodes is not None, ' '.join(sent.words)
        rs_ccg = BobcatParser._build_ccgtree(nodes_to_tree(nodes))
        assert rs_ccg == py_ccg, ' '.join(sent.words)
