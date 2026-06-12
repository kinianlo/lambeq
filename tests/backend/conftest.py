import pytest

from lambeq.backend import fast


@pytest.fixture(autouse=True)
def _validation_on():
    fast.set_validation(True)
    yield
    fast.set_validation(False)
