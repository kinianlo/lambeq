from lambeq import AtomicType, RemoveCupsRewriter, SpiderAnsatz
from lambeq.backend.fast import FSpiderAnsatz, convert
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
