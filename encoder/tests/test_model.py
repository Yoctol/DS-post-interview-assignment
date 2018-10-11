import pytest

import numpy as np

from ..encoder import Encoder
from ..model import MultiTaskModel
from ..task import BinaryClsTask, MultiClsTask, AutoEncoderTask


@pytest.fixture(scope='module')
def multi_task_model():
    encoder = Encoder(input_dim=10, output_dim=5)
    return MultiTaskModel(encoder=encoder)


@pytest.fixture(scope='module')
def tasks_and_data():
    bct = BinaryClsTask('binary_cls_test', 1)
    mct = MultiClsTask('multi_cls_test', 1)
    aet = AutoEncoderTask('auto_encoder_test', 10)

    bct_data = (
        np.random.rand(20, 10),
        np.random.rand(20, 1),
    )
    mct_data = (
        np.random.rand(30, 10),
        np.random.rand(30, 1),
    )
    aet_data = np.random.rand(40, 10)
    return dict([
        (bct, bct_data),
        (mct, mct_data),
        (aet, aet_data),
    ]),


def test_add_task(multi_task_model, tasks_and_data):
    for task in tasks_and_data.keys():
        multi_task_model.add_task(task)

    with pytest.raises(RuntimeError):
        multi_task_model.add_task(task)


def test_evaluate(multi_task_model, tasks_and_data):
    for task, data in tasks_and_data.items():
        loss = multi_task_model.evaluate(task, data)
        assert loss.shape == ()


def test_fit(multi_task_model, tasks_and_data):
    supervised_data = {
        task: data for task, data in tasks_and_data
        if isinstance(data, tuple)
    }
    unsupervised_data = {
        task: data for task, data in tasks_and_data
        if not isinstance(data, tuple)
    }
    multi_task_model.fit(
        supervised_data=supervised_data,
        unsupervised_data=unsupervised_data,
    )
