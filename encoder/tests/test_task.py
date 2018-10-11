import pytest

import numpy as np

from ..task import (
    BinaryClsTask,
    MultiClsTask,
    AutoEncoderTask,
)


@pytest.fixture(scope="module")
def mismatch_xy():
    return (np.random.rand(3, 5), np.random.rand(4, 5))


@pytest.fixture(scope="module")
def match_xy():
    return (np.random.rand(3, 5), np.random.rand(3, 5))


@pytest.fixture(scope="module")
def rank3_x():
    return np.random.rand(3, 3, 3)


def test_binary_cls_task(mismatch_xy, match_xy):
    bct = BinaryClsTask('test', 5)
    with pytest.raises(ValueError):
        bct.validate_data(mismatch_xy)
    bct.validate_data(match_xy)


def test_multi_cls_task(mismatch_xy, match_xy):
    mct = MultiClsTask('test', 5)
    with pytest.raises(ValueError):
        mct.validate_data(mismatch_xy)
    mct.validate_data(match_xy)


def test_auto_encoder_task(rank3_x):
    aet = AutoEncoderTask('test', 5)
    with pytest.raises(ValueError):
        invalid_data = np.random.rand(3, 10)
        aet.validate_data(invalid_data)
    with pytest.raises(ValueError):
        aet.validate_data(rank3_x)

    valid_data = np.random.rand(3, 5)
    aet.validate_data(valid_data)
