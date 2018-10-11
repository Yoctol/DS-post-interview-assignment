import pytest
import tempfile
import shutil

import numpy as np

from encoder import Encoder
from encoder.model import MultiTaskModel
from encoder.task import (
    BinaryClsTask,
    MultiClsTask,
    AutoEncoderTask,
)


def fake_binary_cls_mapping(x):
    return (x[:, 0:1] > 0.5).astype(np.int32)


def fake_multi_cls_mapping(x, n_class):
    return np.expand_dims(np.argmax(x, axis=1) % n_class, axis=-1)


class TestMultiTaskEncoder:
    input_dim = 200
    latent_dim = 20

    @pytest.fixture(scope="module")
    def multi_task_model(self):
        encoder = Encoder(
            input_dim=self.input_dim,
            output_dim=self.latent_dim,
        )
        return MultiTaskModel(encoder=encoder)

    @pytest.fixture(scope="module")
    def tasks_and_data(self):
        bct = BinaryClsTask('binary_cls_test', 1)
        mct = MultiClsTask('multi_cls_test', 10)
        aet = AutoEncoderTask('auto_encoder_test', self.input_dim)

        x = np.random.randn(40, self.input_dim)
        x_b = x[:30]
        x_m = x[30:]
        bct_data = x_b, fake_binary_cls_mapping(x_b)
        mct_data = x_m, fake_multi_cls_mapping(x_m, mct.n_classes)
        aet_data = x

        return dict([
            (bct, bct_data),
            (mct, mct_data),
            (aet, aet_data),
        ])

    def test_add_task(self, multi_task_model, tasks_and_data):
        for task in tasks_and_data.keys():
            multi_task_model.add_task(task)
        # with pytest.raises(RuntimeError):
        #     multi_task_model.add_task(task)

    def test_evaluate(self, multi_task_model, tasks_and_data):
        for task, data in tasks_and_data.items():
            loss = multi_task_model.evaluate(task, data)
            assert loss.shape == ()

    def test_fit(self, multi_task_model, tasks_and_data):
        losses = np.sum([
            multi_task_model.evaluate(task, data)
            for task, data in tasks_and_data.items()
        ])
        supervised_data = {
            task: data for task, data in tasks_and_data.items()
            if isinstance(data, tuple)
        }
        unsupervised_data = {
            task: data for task, data in tasks_and_data.items()
            if not isinstance(data, tuple)
        }
        multi_task_model.fit(
            supervised_data=supervised_data,
            unsupervised_data=unsupervised_data,
        )
        new_losses = np.sum([
            multi_task_model.evaluate(task, data)
            for task, data in tasks_and_data.items()
        ])
        assert new_losses < losses

    def test_encoder_encode(self, multi_task_model):
        encoder = multi_task_model.encoder
        x = np.random.randn(100, self.input_dim)
        y = encoder.encode(x)
        assert y.shape == (100, self.latent_dim)

    def test_encoder_save_load(self, multi_task_model):
        encoder = multi_task_model.encoder
        path = tempfile.mkdtemp()

        encoder.save(path=path)
        loaded = Encoder.load(path=path)

        x = np.random.randn(100, self.input_dim)
        assert encoder.input_dim == loaded.input_dim
        assert encoder.output_dim == loaded.output_dim
        np.testing.assert_array_almost_equal(
            encoder.encode(x),
            loaded.encode(x),
        )
        shutil.rmtree(path)
