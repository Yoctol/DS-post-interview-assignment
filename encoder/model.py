from typing import Dict, Union

import numpy as np

from .task import Task, SupervisedTask, UnsupervisedTask
from .types import (
    Data,
    SupervisedData,
    UnsupervisedData,
)


MultiSupervisedData = Dict[SupervisedTask, SupervisedData]
MultiUnsupervisedData = Dict[UnsupervisedTask, UnsupervisedData]
MultiTaskData = Union[MultiSupervisedData, MultiUnsupervisedData]


class MultiTaskModel:
    def __init__(self, encoder):
        self.encoder = encoder
        self._task = set()
        self.graph = encoder.graph
        self.sess = encoder.sess

    def add_task(self, task: Task):
        if task in self._task:
            raise RuntimeError("Task already exists!")
        self._task.add(task)
        task.build_graph(self.encoder)

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
        # Try to minimize the losses of each tasks

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
        # calculate the loss of given task on given data.
        # output should be np.array of shape ()
