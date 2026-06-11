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
    return BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                        parser_backend='python')


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


def test_parse_batch_parallel_matches_serial(bobcat_parser, rs_parser):
    out = _tagged(bobcat_parser, SENTENCES * 8)
    inputs = _rust_inputs(bobcat_parser, out)
    serial = rs_parser.parse_batch(inputs, 1)
    parallel = rs_parser.parse_batch(inputs, 0)   # 0 = all cores
    assert serial == parallel


def _raw_configured_parser(grammar_data, bobcat_parser):
    rs = bobcat_rs.RustChartParser(
        grammar_data['categories'],
        [tuple(r) for r in grammar_data['binary_rules']],
        [tuple(r) for r in grammar_data['type_changing_rules']],
        [tuple(r) for r in grammar_data['type_raising_rules']],
        bobcat_parser.tagger.model.config.cats,
        None, True, 50000, 32, 1.0, 0.01, 1e-05)
    t = bobcat_parser.tagger
    rs.configure_raw(t.model.config.tags,
                     t.tag_prob_threshold, t.tag_prob_threshold_strategy,
                     t.span_prob_threshold, t.span_prob_threshold_strategy)
    return rs


def test_parse_batch_raw_matches_classic_lane(grammar_data, bobcat_parser,
                                              rs_parser):
    rs_raw = _raw_configured_parser(grammar_data, bobcat_parser)
    out = _tagged(bobcat_parser, SENTENCES)
    classic = rs_parser.parse_batch(_rust_inputs(bobcat_parser, out))
    words = [s.split() for s in SENTENCES]
    raw = bobcat_parser.tagger.forward_topk(words)
    raw_results = rs_raw.parse_batch_raw(
        words, raw[0].numpy(), raw[1].numpy(),
        raw[2].numpy(), raw[3].numpy())
    assert raw_results == classic


def test_parse_batch_raw_requires_configuration(grammar_data,
                                                bobcat_parser, rs_parser):
    import numpy as np
    z3f = np.zeros((1, 1, 1), dtype=np.float32)
    z3i = np.zeros((1, 1, 1), dtype=np.int64)
    with pytest.raises(ValueError):
        rs_parser.parse_batch_raw([['a']], z3f, z3i, z3f, z3i)


def test_fused_lane_in_sentences2trees(bobcat_parser):
    from lambeq import BobcatParser, VerbosityLevel
    rust_parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                               parser_backend='rust')
    assert (rust_parser.sentences2trees(
                SENTENCES, verbose=VerbosityLevel.SUPPRESS.value)
            == bobcat_parser.sentences2trees(
                SENTENCES, verbose=VerbosityLevel.SUPPRESS.value))


@pytest.mark.parametrize('tag_p,tag_strat,span_p,span_strat', [
    (0.002, 'relative', 0.0003, 'relative'),   # shipped config
    (0.01, 'absolute', 0.001, 'absolute'),
    (0, 'relative', 0, 'relative'),            # keep-all
    (1, 'relative', 1, 'relative'),            # top-score-only
])
def test_raw_thresholding_matches_extract_topk(grammar_data, bobcat_parser,
                                               tag_p, tag_strat,
                                               span_p, span_strat):
    """Random logits, all threshold strategies: classic assembly
    (extract_topk + Sentence tuples -> parse_batch) must equal the raw
    lane (topk tensors -> parse_batch_raw) tree-for-tree."""
    import torch
    from lambeq.bobcat.tagger import chart_size, extract_topk, idx2span
    tagger = bobcat_parser.tagger
    n_tags = len(tagger.model.config.tags)
    n_cats = len(tagger.model.config.cats)
    words = [['w%d' % j for j in range(n)] for n in (3, 7, 12)]
    W = max(len(w) for w in words)
    S = chart_size(W)
    torch.manual_seed(42)
    tag_logits = torch.randn(len(words), W, n_tags)
    span_logits = torch.randn(len(words), S, n_cats)

    rs = bobcat_rs.RustChartParser(
        grammar_data['categories'],
        [tuple(r) for r in grammar_data['binary_rules']],
        [tuple(r) for r in grammar_data['type_changing_rules']],
        [tuple(r) for r in grammar_data['type_raising_rules']],
        bobcat_parser.tagger.model.config.cats,
        None, True, 50000, 32, 1.0, 0.01, 1e-05)
    rs.configure_raw(tagger.model.config.tags,
                     tag_p, tag_strat, span_p, span_strat)

    # classic assembly (mirrors Tagger.parse + _prepare_sentence)
    tags_list = tagger.model.config.tags
    tag_out = extract_topk(tag_logits, [len(w) for w in words],
                           tagger.tag_top_k, tag_p, tag_strat, False)
    span_out = extract_topk(span_logits,
                            [chart_size(len(w)) for w in words],
                            tagger.span_top_k, span_p, span_strat, True)
    classic_inputs = []
    for w, t_rows, s_rows in zip(words, tag_out, span_out):
        supertags = [[(tags_list[i], sc) for i, sc in row] for row in t_rows]
        spans = {idx2span(i): {ci: cs for ci, cs in row}
                 for i, row in enumerate(s_rows) if row}
        classic_inputs.append((w, supertags, spans))
    classic = rs.parse_batch(classic_inputs)

    def topk(logits, k):
        s, i = logits.float().log_softmax(-1).topk(k)
        return s.cpu().contiguous(), i.cpu().contiguous()

    ts, ti = topk(tag_logits, tagger.tag_top_k)
    ss, si = topk(span_logits, tagger.span_top_k)
    raw = rs.parse_batch_raw(words, ts.numpy(), ti.numpy(),
                             ss.numpy(), si.numpy())
    assert raw == classic
