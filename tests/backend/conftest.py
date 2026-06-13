import pytest

from lambeq.backend import fast


@pytest.fixture(autouse=True)
def _validation_on():
    fast.set_validation(True)
    yield
    fast.set_validation(False)


@pytest.fixture(scope='module')
def bobcat_trees():
    from lambeq import BobcatParser, VerbosityLevel
    parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                          parser_backend='rust')
    with open('/tmp/coco_bench.txt') as f:
        sents = [line.split() for line in f if line.strip()][:40]
    trees = parser.sentences2trees(sents, tokenised=True,
                                   suppress_exceptions=True)
    return [t for t in trees if t is not None]


@pytest.fixture(scope='module')
def bobcat_diagrams(bobcat_trees):
    return [t.to_diagram() for t in bobcat_trees]
