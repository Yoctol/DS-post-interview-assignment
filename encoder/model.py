import numpy as np
import tensorflow as tf

from .task import Task
from .types import (
    Data,
    MultiSupervisedData,
    MultiUnsupervisedData,
    MultiTaskData,
)


class MultiTaskModel:
    def __init__(self, encoder):
        self.encoder = encoder
        self._task = set()
        self.graph = encoder.graph

    def add_task(self, task: Task):
        if task in self._task:
            raise RuntimeError("Task already exists!")
        self._task.add(task)
        with tf.variable_scope(task.name), self.graph.as_default():
            # TODO
            # Please feel free to define the arguments of build_graph
            task.build_graph()

    def fit(
            self,
            supervised_data: MultiSupervisedData,
            unsupervised_data: MultiUnsupervisedData,
        ):
        if not supervised_data and not unsupervised_data:
            raise RuntimeError()
        self._validate_multi_task_data(supervised_data)
        self._validate_multi_task_data(unsupervised_data)
        # TODO

    def _validate_multi_task_data(self, multi_task_data: MultiTaskData):
        for task, data in multi_task_data.items():
            self._validate_data(task, data)

    def _validate_data(self, task: Task, data: Data):
        if task not in self._task:
            raise KeyError(f"Unregistered task: {task}.")
        self.encoder.validate_data(data)
        task.validate_data(data)

    def evaluate(self, task: Task, data: Data) -> np.ndarray:
        self._validate_data(task, data)
        # TODO
