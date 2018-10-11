import pytest

# import numpy as np

from ..encoder import Encoder
from ..model import MultiTaskModel
from ..task import BinaryClsTask, MultiClsTask, AutoEncoderTask


@pytest.fixture(scope='module')
def multi_task_model():
    encoder = Encoder(10, 5)
    return MultiTaskModel(encoder=encoder)


@pytest.fixture(scope='module')
def tasks():
    bct = BinaryClsTask('binary_cls_test', 1)
    mct = MultiClsTask('multi_cls_test', 1)
    aet = AutoEncoderTask('auto_encoder_test', 5)
    return [bct, mct, aet]


def test_add_task(multi_task_model, tasks):
    for task in tasks:
        multi_task_model.add_task(task)

    with pytest.raises(RuntimeError):
        multi_task_model.add_task(tasks[-1])


def test_evaluate(multi_task_model, tasks):
    for task in tasks:
        loss = multi_task_model.evaluate(task)
        assert loss.shape == ()


def test_fit(multi_task_model):
    pass
