import abc


class Task(abc.ABC):
    def __init__(self, name, output_dim):
        self.name = name
        self.output_dim = output_dim

    @abc.abstractmethod
    def build_graph(self):
        pass

    def __hash__(self):
        return self.name.__hash__()

    @abc.abstractmethod
    def validate_data(self, data):
        pass


class BinaryClsTask(Task):
    def build_graph(self):
        pass

    def validate_data(self, data):
        if len(data) != 2:
            raise RuntimeError()
        x, y = data
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise RuntimeError()
        if x.shape[0] != y.shape[0]:
            raise RuntimeError()
        if y.shape[1] != self.output_dim:
            raise RuntimeError()


class MultiClsTask(Task):
    def build_graph(self):
        pass


class AutoEncoderTask(Task):
    def build_graph(self):
        pass
