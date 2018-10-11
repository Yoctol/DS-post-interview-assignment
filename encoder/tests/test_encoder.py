import pytest
import tempfile

import numpy as np

from ..encoder import Encoder


@pytest.fixture(scope="module")
def encoder():
    return Encoder(input_dim=10, output_dim=2)


def test_encode_shape(encoder):
    with pytest.raises(ValueError):
        invalid_x = np.random.rand(5, 4)
        encoder.encode(invalid_x)

    n_samples = 20
    x = np.random.rand(n_samples, encoder.input_dim)
    y = encoder.encode(x)
    assert y.shape == (n_samples, encoder.output_dim)


def test_save_load(encoder):
    with tempfile.NamedTemporaryFile() as t:
        encoder.save(path=t.name)
        loaded = Encoder.load(path=t.name)

    assert encoder.input_dim == loaded.input_dim
    assert encoder.output_dim == loaded.output_dim
