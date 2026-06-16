import pytest

from lambeq import AtomicType, RemoveCupsRewriter, SpiderAnsatz
from lambeq.backend.fast import FSpiderAnsatz, compile_fast_circuits, convert
from lambeq.backend.fast.model import FastPytorchModel
from lambeq.backend.tensor import Dim

OB = {t: Dim(2) for t in AtomicType}


def test_from_fast_diagrams_symbols_match(bobcat_diagrams):
    from lambeq import PytorchModel
    rc = RemoveCupsRewriter()
    ds = bobcat_diagrams[:10]
    legacy = PytorchModel.from_diagrams([SpiderAnsatz(OB)(rc(d)) for d in ds])
    fans = FSpiderAnsatz(OB)
    fast = FastPytorchModel.from_fast_diagrams(
        [fans(convert.to_fast(rc(d))) for d in ds])
    # same symbol SET and same ORDER (weights indexed positionally)
    assert [s.name for s in fast.symbols] == [s.name for s in legacy.symbols]


def test_spec_for_accepts_fdiagram(bobcat_diagrams):
    fans = FSpiderAnsatz(OB)
    fd = fans(convert.to_fast(bobcat_diagrams[0]))
    m = FastPytorchModel.from_fast_diagrams([fd])
    # must NOT call convert.to_fast on an FDiagram
    spec = m._spec_for(fd)
    assert spec is not None


def test_pipeline_matches_legacy_end_to_end(bobcat_trees):
    torch = pytest.importorskip('torch')
    from collections import Counter
    from lambeq import PytorchModel

    rc = RemoveCupsRewriter()
    diagrams = [t.to_diagram() for t in bobcat_trees]
    g_circuits = [SpiderAnsatz(OB)(rc(d)) for d in diagrams]
    # filter to uniform output shape so PytorchModel can stack
    shapes = [tuple(c.cod.dim) for c in g_circuits]
    modal = Counter(shapes).most_common(1)[0][0]
    keep = [i for i, s in enumerate(shapes) if s == modal]
    trees = [bobcat_trees[i] for i in keep]
    g_circuits = [g_circuits[i] for i in keep]
    assert len(keep) >= 4

    legacy = PytorchModel.from_diagrams(g_circuits)
    torch.manual_seed(0)
    legacy.initialise_weights()
    expected = legacy.get_diagram_output(g_circuits)

    f_circuits = compile_fast_circuits(trees, OB)
    fast = FastPytorchModel.from_fast_diagrams(f_circuits)
    fast.symbols = legacy.symbols
    fast.weights = legacy.weights          # share exact weights
    got = fast.get_diagram_output(f_circuits)
    assert torch.allclose(got, expected, atol=1e-5)
