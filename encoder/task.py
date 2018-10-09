import abc

import numpy as np


class Task(abc.ABC):
    def __init__(self, name, output_dim):
        self.name = name
        self.output_dim = output_dim

    @abc.abstractmethod
    def build_graph(self, graph):
        pass

    def __hash__(self):
        return self.name.__hash__()

    @abc.abstractmethod
    def validate_data(self, data):
        pass


class SupervisedTask(Task):
    def validate_data(self, data):
        if len(data) != 2:
            raise RuntimeError("Data should be (x, y) pairs!")
        x, y = data
        if len(x.shape) != 2:
            raise RuntimeError("Input data should be rank 2!")
        if x.shape[0] != y.shape[0]:
            raise RuntimeError("Data pairs should have same length!")
        if len(y.shape) != 2:
            raise RuntimeError("Output data should be rank 2!")
        if y.shape[1] != self.output_dim:
            raise RuntimeError(f"Invalid output data dimension: {y.shape[1]} != {self.output_dim}!")


class BinaryClsTask(SupervisedTask):
    def build_graph(self, graph):
        pass


class MultiClsTask(SupervisedTask):
    def build_graph(self, graph):
        pass


class UnsupervisedTask(Task):
    def validate_data(self, data):
        if not isinstance(data, np.ndarray):
            raise RuntimeError("Output data should be numpy.array!")
        if len(data.shape) != 2:
            raise RuntimeError("Output data should be rank 2!")
        if data.shape[1] != self.output_dim:
            raise RuntimeError(
                f"Invalid output data dimension: {data.shape[1]} != {self.output_dim}!")


class AutoEncoderTask(UnsupervisedTask):
    def build_graph(self, graph):
        pass
