# tests/backend/test_fast_training.py
import pytest

torch = pytest.importorskip('torch')

from lambeq import (AtomicType, BobcatParser, PytorchModel,  # noqa: E402
                    RemoveCupsRewriter, SpiderAnsatz, VerbosityLevel)
from lambeq.backend.fast.model import FastPytorchModel  # noqa: E402
from lambeq.backend.tensor import Dim  # noqa: E402

KEYWORDS = {'man', 'woman', 'men', 'women', 'people', 'person', 'boy',
            'girl', 'child', 'dog', 'cat', 'horse', 'bird', 'animal'}


def _label(tokens):
    return int(any(t.lower() in KEYWORDS for t in tokens))


def _build_circuits(token_lists, dim=2):
    parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value,
                          parser_backend='rust')
    trees = parser.sentences2trees(token_lists, tokenised=True,
                                   suppress_exceptions=True)
    ansatz = SpiderAnsatz({t: Dim(dim) for t in AtomicType})
    remove_cups = RemoveCupsRewriter()
    circuits, labels = [], []
    for toks, tree in zip(token_lists, trees):
        if tree is None:
            continue
        circuits.append(ansatz(remove_cups(tree.to_diagram())))
        labels.append(_label(toks))
    return circuits, labels


def _filter_uniform_shape(circuits, labels):
    from collections import Counter
    shapes = [tuple(c.cod.dim) for c in circuits]
    modal = Counter(shapes).most_common(1)[0][0]
    keep = [i for i, s in enumerate(shapes) if s == modal]
    return ([circuits[i] for i in keep], [labels[i] for i in keep], modal)


def _train(model, circuits, labels, epochs, lr):
    onehot = torch.nn.functional.one_hot(
        torch.tensor(labels), num_classes=2).float()
    opt = torch.optim.Adam(model.weights, lr=lr)
    model.get_diagram_output(circuits)          # warm up / build spec cache
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        out = model.get_diagram_output(circuits)
        loss = ((out - onehot) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_fast_model_training_trajectory_matches():
    with open('/tmp/coco_bench.txt') as f:
        token_lists = [line.split() for line in f if line.strip()][:30]
    circuits, labels = _build_circuits(token_lists)
    circuits, labels, shape = _filter_uniform_shape(circuits, labels)
    circuits, labels = circuits[:12], labels[:12]
    assert len(circuits) >= 4 and shape == (2,)

    old = PytorchModel.from_diagrams(circuits)
    torch.manual_seed(0)
    old.initialise_weights()

    fast = FastPytorchModel.from_diagrams(circuits)
    fast.symbols = old.symbols
    fast.weights = torch.nn.ParameterList(
        [torch.nn.Parameter(w.detach().clone()) for w in old.weights])

    old_losses = _train(old, circuits, labels, epochs=3, lr=0.05)
    fast_losses = _train(fast, circuits, labels, epochs=3, lr=0.05)

    assert torch.allclose(torch.tensor(old_losses),
                          torch.tensor(fast_losses), atol=1e-4), \
        f'old={old_losses} fast={fast_losses}'
