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


class BinaryClsTask(Task):
    # TODO
    pass


class ClsTask(Task):
    # TODO
    pass


class AutoEncoderTask(Task):
    # TODO
    pass
