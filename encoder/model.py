from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from .task import Task

SupervisedData = Tuple[np.ndarray, np.ndarray]
MultiSupervisedData = Dict[Task, SupervisedData]
MultiUnsupervisedData = Dict[Task, np.ndarray]


class MultiTaskModel:
    def __init__(self, encoder):
        self.encoder = encoder
        self._task = set()
        self.graph = encoder.graph

    def add_task(self, task):
        if task in self._task:
            raise RuntimeError("Task already exists!")
        self._task.add(task)
        with tf.variable_scope(task.name), self.graph.as_default():
            task.build_graph()

    def fit(
            self,
            supervised: MultiSupervisedData,
            unsupervised: MultiUnsupervisedData,
        ):
        if not supervised and not unsupervised:
            raise RuntimeError()
        self._validate_data(supervised)
        self._validate_data(unsupervised)

    def _validate_data(self, multi_task_data):
        for task, data in multi_task_data.items():
            if task not in self._task:
                raise RuntimeError()
            self.encoder.validate_data(data)
            task.validate_data(data)

    def evaluate(self):
        pass
