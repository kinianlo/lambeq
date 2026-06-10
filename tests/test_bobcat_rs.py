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
