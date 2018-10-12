import pytest
import tempfile
import shutil

import numpy as np

from encoder import Encoder
from encoder.model import MultiTaskModel
from encoder.task import (
    MultiLabelTask,
    MultiClassTask,
    AutoEncoderTask,
)


def fake_multi_label_mapping(x, n_labels):
    return (x[:, 0: n_labels] > 0.5).astype(np.int32)


def fake_multi_class_mapping(x, n_class):
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
    def supervised_tasks_and_data(self):
        mlt = MultiLabelTask('multi_label_test', 5)
        mct = MultiClassTask('multi_class_test', 10)

        x = np.random.randn(40, self.input_dim)
        x_mlt = x[:30]
        x_mct = x[30:]
        mlt_data = x_mlt, fake_multi_label_mapping(x_mlt, mlt.n_labels)
        mct_data = x_mct, fake_multi_class_mapping(x_mct, mct.n_classes)

        return {
            mlt: mlt_data,
            mct: mct_data,
        }

    @pytest.fixture(scope="module")
    def unsupervised_tasks_and_data(self):
        x = np.random.randn(100, self.input_dim)
        aet = AutoEncoderTask('auto_encoder_test', self.input_dim)
        aet_data = x
        return {aet: aet_data}

    @pytest.fixture(scope="module")
    def tasks_and_data(supervised_tasks_and_data, unsupervised_tasks_and_data):
        tasks = supervised_tasks_and_data.copy()
        tasks.update(unsupervised_tasks_and_data)
        return tasks

    def test_add_task(self, multi_task_model, tasks_and_data):
        for task in tasks_and_data.keys():
            multi_task_model.add_task(task)
        with pytest.raises(RuntimeError):
            multi_task_model.add_task(task)

    def test_evaluate(self, multi_task_model, tasks_and_data):
        for task, data in tasks_and_data.items():
            loss = multi_task_model.evaluate(task, data)
            assert loss.shape == ()

    def evaluate_on_tasks(self, multi_task_model, tasks_and_data):
        return [
            multi_task_model.evaluate(task, data)
            for task, data in tasks_and_data]

    def test_fit(
            self,
            multi_task_model,
            supervised_tasks_and_data,
            unsupervised_tasks_and_data,
            tasks_and_data,
        ):
        original_losses = self.evaluate_on_tasks(multi_task_model, tasks_and_data)
        multi_task_model.fit(
            supervised_data=supervised_tasks_and_data,
            unsupervised_data=unsupervised_tasks_and_data,
        )
        new_losses = self.evaluate_on_tasks(multi_task_model, tasks_and_data)
        assert all(new_losses < original_losses)

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
            decimal=6,
        )
        shutil.rmtree(path)
