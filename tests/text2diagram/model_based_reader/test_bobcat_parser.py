import pytest
import torch

from io import StringIO
from unittest.mock import patch

from lambeq import BobcatParseError, BobcatParser, CCGType, VerbosityLevel
from lambeq.backend.grammar import Cup, Diagram, Ty, Word
from lambeq.bobcat.tagger import Tagger, chart_size


@pytest.fixture(scope='module')
def bobcat_parser():
    return BobcatParser(verbose=VerbosityLevel.SUPPRESS.value)


@pytest.fixture
def sentence():
    return 'What Alice is and is not .'


@pytest.fixture
def tokenised_sentence():
    return ['What', 'Alice', 'is', 'and', 'is', 'not', '.']



@pytest.fixture
def simple_diagram():
    n, s = map(Ty, 'ns')
    return Diagram.create_pregroup_diagram(
        words=[Word('Alice', n), Word('likes', n.r @ s @ n.l), Word('Bob', n)],
        morphisms=[(Cup, 3, 4),(Cup, 0, 1)]
    )


@pytest.fixture
def simple_diagram_w_fullstop():
    n, s = map(Ty, 'ns')
    return Diagram.create_pregroup_diagram(
        words=[Word('Alice', n), Word('likes', n.r @ s @ n.l), Word('Bob.', n)],
        morphisms=[(Cup, 3, 4),(Cup, 0, 1)]
    )


@pytest.fixture
def tokenised_empty_sentence():
    return []


def test_sentence2diagram(bobcat_parser, sentence, simple_diagram, simple_diagram_w_fullstop):
    assert bobcat_parser.sentence2diagram(sentence) is not None

    assert bobcat_parser.sentence2diagram('Alice likes Bob') == simple_diagram
    assert bobcat_parser.sentence2diagram('Alice likes Bob .') == simple_diagram
    assert bobcat_parser.sentence2diagram('Alice likes Bob.') == simple_diagram_w_fullstop


def test_sentence2tree(bobcat_parser, sentence):
    assert bobcat_parser.sentence2tree(sentence) is not None


def test_empty_sentences(bobcat_parser):
    with pytest.raises(ValueError):
        bobcat_parser.sentence2tree('')
    assert bobcat_parser.sentence2tree('', suppress_exceptions=True) is None

    with pytest.raises(ValueError):
        bobcat_parser.sentence2tree('   ')
    assert bobcat_parser.sentence2tree('   ', suppress_exceptions=True) is None


def test_tokenised_empty_sentences(bobcat_parser, tokenised_empty_sentence):
    with pytest.raises(ValueError):
        bobcat_parser.sentence2tree(tokenised_empty_sentence, tokenised=True)
    assert bobcat_parser.sentence2tree(
        tokenised_empty_sentence,
        tokenised=True,
        suppress_exceptions=True
    ) is None


def test_failed_sentence(bobcat_parser):
    def fail(*args, **kwargs):
        raise Exception

    old_parser = bobcat_parser.parser
    bobcat_parser.parser = fail

    try:
        with pytest.raises(BobcatParseError):
            bobcat_parser.sentence2tree('a')
        assert bobcat_parser.sentence2tree('a', suppress_exceptions=True) is None
    finally:
        bobcat_parser.parser = old_parser


def test_sentence2tree_tokenised(bobcat_parser, tokenised_sentence):
    assert bobcat_parser.sentence2tree(tokenised_sentence, tokenised=True) is not None


def test_sentences2diagrams(bobcat_parser, sentence, simple_diagram, simple_diagram_w_fullstop):
    assert bobcat_parser.sentences2diagrams([sentence]) is not None

    assert bobcat_parser.sentences2diagrams(['Alice likes Bob']) == [simple_diagram]
    assert bobcat_parser.sentences2diagrams(['Alice likes Bob .']) == [simple_diagram]
    assert bobcat_parser.sentences2diagrams(['Alice likes Bob.']) == [simple_diagram_w_fullstop]


def test_sentence2diagram_tokenised(bobcat_parser, tokenised_sentence, simple_diagram, simple_diagram_w_fullstop):
    assert bobcat_parser.sentence2diagram(tokenised_sentence, tokenised=True) is not None

    assert bobcat_parser.sentence2diagram(
        'Alice likes Bob'.split(), tokenised=True
    ) == simple_diagram
    assert bobcat_parser.sentence2diagram(
        'Alice likes Bob .'.split(), tokenised=True
    ) == simple_diagram
    assert bobcat_parser.sentence2diagram(
        'Alice likes Bob.'.split(), tokenised=True
    ) == simple_diagram_w_fullstop


def test_sentences2diagrams_tokenised(bobcat_parser, tokenised_sentence, simple_diagram, simple_diagram_w_fullstop):
    assert bobcat_parser.sentences2diagrams([tokenised_sentence], tokenised=True) is not None

    assert bobcat_parser.sentences2diagrams(
        ['Alice likes Bob'.split()], tokenised=True
    ) == [simple_diagram]
    assert bobcat_parser.sentences2diagrams(
        ['Alice likes Bob .'.split()], tokenised=True
    ) == [simple_diagram]
    assert bobcat_parser.sentences2diagrams(
        ['Alice likes Bob.'.split()], tokenised=True
    ) == [simple_diagram_w_fullstop]


def test_tokenised_type_check_untokenised_sentence(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentence2diagram(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentence2diagram(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2diagrams([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2diagrams([tokenised_sentence], tokenised=False)


def test_tokenised_type_check_untokenised_sentence_s2t(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentence2tree(sentence, tokenised=True)


def test_tokenised_type_check_tokenised_sentence_s2t(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentence2tree(tokenised_sentence, tokenised=False)


def test_tokenised_type_check_untokenised_batch_s2t(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2trees([sentence], tokenised=True)


def test_tokenised_type_check_tokenised_batch_s2t(bobcat_parser, tokenised_sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2trees([tokenised_sentence], tokenised=False)


def test_verbosity_exceptions_init():
    with pytest.raises(ValueError):
        bobcatbank_parser = BobcatParser(verbose='invalid_option')


def test_kwargs_exceptions_init():
    with pytest.raises(TypeError):
        bobcatbank_parser = BobcatParser(nonexisting_arg='invalid_option')


def test_verbosity_exceptions_sentences2trees(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        _ = bobcat_parser.sentences2trees([sentence], verbose='invalid_option')


def test_text_progress(bobcat_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _ = bobcat_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.TEXT.value)
        assert fake_out.getvalue().rstrip() == 'Tagging sentences.\nParsing tagged sentences.\nTurning parse trees to diagrams.'


def test_tqdm_progress(bobcat_parser, sentence):
    with patch('sys.stderr', new=StringIO()) as fake_out:
        _ = bobcat_parser.sentences2diagrams([sentence], verbose=VerbosityLevel.PROGRESS.value)
        assert fake_out.getvalue().rstrip() != ''


def test_root_filtering(bobcat_parser):
    S = CCGType.SENTENCE
    N = CCGType.NOUN_PHRASE

    sentence1 = 'do'
    sentence2 = 'I do'
    assert bobcat_parser.sentence2tree(sentence1).biclosed_type == N >> S
    assert bobcat_parser.sentence2tree(sentence2).biclosed_type == S

    bobcat_parser.parser.set_root_cats(['NP'])
    try:
        assert bobcat_parser.sentence2tree(sentence1).biclosed_type == N
        assert bobcat_parser.sentence2tree(sentence2).biclosed_type == N
    finally:
        bobcat_parser.parser.set_root_cats(None)


def test_tagger_uses_inference_mode(bobcat_parser):
    recorded = []
    handle = bobcat_parser.tagger.model.register_forward_pre_hook(
        lambda module, args: recorded.append(
            torch.is_inference_mode_enabled()))
    try:
        bobcat_parser.tagger.parse([['Alice', 'likes', 'Bob']])
    finally:
        handle.remove()
    assert recorded == [True]


def test_make_batches_sorts_by_length(bobcat_parser):
    sentences = [['a'] * n for n in (5, 1, 3, 2, 4)]
    batches = bobcat_parser.tagger.make_batches(sentences, batch_size=2)
    assert batches == [[1, 3], [2, 4], [0]]


def test_tagger_restores_input_order(bobcat_parser):
    sentences = [
        'Alice likes Bob and Claire likes Dave'.split(),
        'Alice likes Bob'.split(),
        'I do'.split(),
        'What Alice is and is not .'.split(),
    ]
    output = bobcat_parser.tagger(sentences, batch_size=2,
                                  verbose=VerbosityLevel.SUPPRESS.value)
    assert [s.words for s in output.sentences] == sentences


def test_tagger_invalid_batch_size_override(bobcat_parser):
    with pytest.raises(ValueError):
        bobcat_parser.tagger([['a']], batch_size=-1)


def test_make_batches_respects_span_budget(bobcat_parser):
    tagger = bobcat_parser.tagger
    tagger.max_spans_per_batch = 20
    try:
        sentences = [['a'] * n for n in (1, 2, 3, 4, 5)]
        # batch_size=1 proves the budget overrides batch_size
        batches = tagger.make_batches(sentences, batch_size=1)
        # every sentence appears exactly once
        assert sorted(i for batch in batches for i in batch) == [0, 1, 2, 3, 4]
        assert any(len(batch) > 1 for batch in batches)
        for batch in batches:
            max_len = max(len(sentences[i]) for i in batch)
            padded_spans = len(batch) * chart_size(max_len)
            assert len(batch) == 1 or padded_spans <= 20

        # exact boundary: 3 sentences of length 3 give
        # 3 * chart_size(3) = 18 padded spans, fitting budget 18 exactly
        tagger.max_spans_per_batch = 18
        assert (tagger.make_batches([['a'] * 3] * 3, batch_size=1)
                == [[0, 1, 2]])
        # one below the boundary forces a split
        tagger.max_spans_per_batch = 17
        assert (tagger.make_batches([['a'] * 3] * 3, batch_size=1)
                == [[0, 1], [2]])

        # a sentence exceeding the budget alone forms a singleton batch
        tagger.max_spans_per_batch = 5
        assert tagger.make_batches([['a'] * 5], batch_size=1) == [[0]]
    finally:
        tagger.max_spans_per_batch = None


def test_invalid_max_spans_per_batch(bobcat_parser):
    with pytest.raises(ValueError):
        Tagger(bobcat_parser.tagger.model,
               bobcat_parser.tagger.tokenizer,
               max_spans_per_batch=0)


def test_invalid_dtype(bobcat_parser):
    with pytest.raises(ValueError):
        Tagger(bobcat_parser.tagger.model,
               bobcat_parser.tagger.tokenizer,
               dtype='float8')


def test_parallel_parsing_matches_serial(bobcat_parser):
    sentences = ['Alice likes Bob',
                 'What Alice is and is not .',
                 'I do not like Bob']
    serial = bobcat_parser.sentences2trees(
        sentences, verbose=VerbosityLevel.SUPPRESS.value)
    parallel = bobcat_parser.sentences2trees(
        sentences, n_jobs=2, verbose=VerbosityLevel.SUPPRESS.value)
    assert parallel == serial


def test_invalid_n_jobs(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        bobcat_parser.sentences2trees([sentence], n_jobs=0)


def test_invalid_n_jobs_non_integer(bobcat_parser, sentence):
    with pytest.raises(ValueError):
        bobcat_parser.sentences2trees([sentence], n_jobs=2.0)


def test_parallel_parsing_with_empty_sentence(bobcat_parser):
    sentences = ['Alice likes Bob', '', 'I do']
    trees = bobcat_parser.sentences2trees(
        sentences, n_jobs=2, suppress_exceptions=True,
        verbose=VerbosityLevel.SUPPRESS.value)
    assert trees[0] is not None
    assert trees[1] is None
    assert trees[2] is not None


def test_reduced_precision_tagging(bobcat_parser, sentence):
    tagger = bobcat_parser.tagger
    assert tagger.dtype is None
    tagger.dtype = 'bfloat16'
    try:
        with patch('lambeq.bobcat.tagger.torch.autocast',
                   wraps=torch.autocast) as autocast_spy:
            tree = bobcat_parser.sentence2tree(sentence)
        assert tree is not None
        autocast_spy.assert_called_once()
        assert autocast_spy.call_args.kwargs['dtype'] is torch.bfloat16
    finally:
        tagger.dtype = None


def test_invalid_parser_backend():
    with pytest.raises(ValueError):
        BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                     parser_backend='cobol')


def test_rust_backend_matches_python(bobcat_parser):
    pytest.importorskip('bobcat_rs')
    from lambeq.bobcat.rust_backend import RustBackend
    rust_parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                               parser_backend='rust')
    assert isinstance(rust_parser.parser, RustBackend)
    sentences = ['Alice likes Bob', 'What Alice is and is not .']
    assert (rust_parser.sentences2trees(
                sentences, verbose=VerbosityLevel.SUPPRESS.value)
            == bobcat_parser.sentences2trees(
                sentences, verbose=VerbosityLevel.SUPPRESS.value))
    # root-cat switching must work on the Rust backend too
    rust_parser.parser.set_root_cats(['NP'])
    try:
        tree = rust_parser.sentence2tree('I do')
        assert tree.biclosed_type == CCGType.NOUN_PHRASE
    finally:
        rust_parser.parser.set_root_cats(None)


def test_gpu_tagger_defaults_helper():
    from lambeq.text2diagram.model_based_reader.bobcat_parser import (
        _apply_gpu_tagger_defaults)
    # CUDA, nothing user-set, shipped config: both defaults apply
    config = {'batch_size': 4}
    _apply_gpu_tagger_defaults(config, 'cuda', set())
    assert config == {'batch_size': 64, 'dtype': 'float16'}
    # explicit user settings always win
    config = {'batch_size': 4, 'dtype': 'bfloat16'}
    _apply_gpu_tagger_defaults(config, 'cuda', {'batch_size', 'dtype'})
    assert config == {'batch_size': 4, 'dtype': 'bfloat16'}
    # non-shipped pipeline batch_size is respected
    config = {'batch_size': 32}
    _apply_gpu_tagger_defaults(config, 'cuda', set())
    assert config['batch_size'] == 32 and config['dtype'] == 'float16'
    # CPU: batch tuned, no dtype change
    config = {'batch_size': 4}
    _apply_gpu_tagger_defaults(config, 'cpu', set())
    assert config == {'batch_size': 16}
    config = {'batch_size': 4}
    _apply_gpu_tagger_defaults(config, 'cpu', {'batch_size'})
    assert config == {'batch_size': 4}


def test_cpu_parser_keeps_classic_defaults(bobcat_parser):
    assert bobcat_parser.tagger.dtype is None
    assert bobcat_parser.tagger.batch_size == 16


def test_compile_model_flag(bobcat_parser):
    with pytest.raises(ValueError):
        BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                     compile_model='yes')
    compiled = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                            compile_model=True)
    assert type(compiled.tagger.model.bert).__name__ == 'OptimizedModule'


def test_invalid_tagger_backend():
    with pytest.raises(ValueError):
        BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                     tagger_backend='tensorflow')


def test_onnx_tagger_matches_torch(bobcat_parser):
    onnxruntime = pytest.importorskip('onnxruntime')
    import pathlib
    import subprocess
    import sys
    model_dir = pathlib.Path.home() / '.cache/lambeq/bobcat/bobcat'
    if not (model_dir / 'bobcat-body.onnx').exists():
        subprocess.run([sys.executable, 'tools/export_onnx.py'],
                       check=True)
    onnx_parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                               tagger_backend='onnx')
    sentences = ['Alice likes Bob', 'I do not like Bob']
    assert (onnx_parser.sentences2trees(
                sentences, verbose=VerbosityLevel.SUPPRESS.value)
            == bobcat_parser.sentences2trees(
                sentences, verbose=VerbosityLevel.SUPPRESS.value))
