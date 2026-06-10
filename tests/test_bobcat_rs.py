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
