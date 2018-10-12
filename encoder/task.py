import abc

import tensorflow as tf

from .types import SupervisedData, UnsupervisedData


class Task(abc.ABC):
    def __init__(self, name: str, output_dim: int):
        self.name = name
        self.output_dim = output_dim

    def build_graph(self, encoder):
        graph = encoder.graph
        with graph.as_default(), tf.variable_scope(self.name) as vs:
            self._build_graph(encoder)
            encoder.sess.run(
                tf.variables_initializer(
                    var_list=graph.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope=vs.name,
                    ),
                )
            )

    @abc.abstractmethod
    def _build_graph(self, encoder):
        pass

    @abc.abstractmethod
    def validate_data(self, data):
        pass

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return f"{self.name}"


class SupervisedTask(Task):
    def validate_data(self, data: SupervisedData):
        x, y = data
        if len(x.shape) != 2:
            raise ValueError("Input data should be rank 2!")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Data pairs should have the same length!")
        if len(y.shape) != 2:
            raise ValueError("Output data should be rank 2!")
        if y.shape[1] != self.output_dim:
            raise ValueError(f"Invalid output data dimension: {y.shape[1]} != {self.output_dim}!")


class MultiLabelTask(SupervisedTask):

    def __init__(self, name: str, n_labels: int):
        super().__init__(name=name, output_dim=n_labels)

    @property
    def n_labels(self):
        return self.output_dim

    def _build_graph(self, encoder):
        # TODO
        pass


class MultiClassTask(SupervisedTask):

    def __init__(self, name: str, n_classes: int):
        super().__init__(name=name, output_dim=1)
        self.n_classes = n_classes

    def _build_graph(self, encoder):
        # TODO
        pass


class UnsupervisedTask(Task):
    def validate_data(self, data: UnsupervisedData):
        if len(data.shape) != 2:
            raise ValueError("Output data should be rank 2!")
        if data.shape[1] != self.output_dim:
            raise ValueError(
                f"Invalid output data dimension: {data.shape[1]} != {self.output_dim}!")


class AutoEncoderTask(UnsupervisedTask):

    def _build_graph(self, encoder):
        # TODO
        pass
