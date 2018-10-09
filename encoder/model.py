import tensorflow as tf

from .task import Task
from .types import MultiSupervisedData, MultiUnsupervisedData


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
        self._validate_data(supervised_data)
        self._validate_data(unsupervised_data)
        # TODO

    def _validate_data(self, multi_task_data):
        for task, data in multi_task_data.items():
            if task not in self._task:
                raise KeyError(f"Unregistered task: {task}.")
            self.encoder.validate_data(data)
            task.validate_data(data)

    def evaluate(self):
        # TODO
        pass
