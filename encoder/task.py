import abc

from .types import SupervisedData, UnsupervisedData


class Task(abc.ABC):
    def __init__(self, name: str, output_dim: int):
        self.name = name
        self.output_dim = output_dim

    @abc.abstractmethod
    def build_graph(self):
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


class BinaryClsTask(SupervisedTask):
    def build_graph(self, *args, **kwargs):
        pass


class MultiClsTask(SupervisedTask):
    def build_graph(self, *args, **kwargs):
        pass


class UnsupervisedTask(Task):
    def validate_data(self, data: UnsupervisedData):
        if len(data.shape) != 2:
            raise ValueError("Output data should be rank 2!")
        if data.shape[1] != self.output_dim:
            raise ValueError(
                f"Invalid output data dimension: {data.shape[1]} != {self.output_dim}!")


class AutoEncoderTask(UnsupervisedTask):
    def build_graph(self, *args, **kwargs):
        pass
