import abc

import tensorflow as tf

from .types import SupervisedData, UnsupervisedData


class Task(abc.ABC):
    def __init__(self, name: str, output_dim: int):
        # TODO
        # This object should support additional model config or hyperparameters
        # with default value.

        # You can develope your own design. (without breaking the interface.)

        self.name = name
        self.output_dim = output_dim

    def extend_encoder_graph(self, encoder):
        graph = encoder.graph
        with graph.as_default(), tf.variable_scope(self.name) as vs:
            self._extend_encode_graph(encoder)
            encoder.sess.run(
                tf.variables_initializer(
                    var_list=graph.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope=vs.name,
                    ),
                )
            )

    @abc.abstractmethod
    def _extend_encode_graph(self, encoder):
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

    def _extend_encode_graph(self, encoder):
        # TODO
        # Build a graph containing necessary operations and tensors
        # to train and predict multi-label data.

        # the labels will be np.array of shape (N, self.n_labels)
        # and np.int32 value in {0, 1}

        # the prediction should be based on the output of encoder.

        pass


class MultiClassTask(SupervisedTask):

    def __init__(self, name: str, n_classes: int):
        super().__init__(name=name, output_dim=1)
        self.n_classes = n_classes

    def _extend_encode_graph(self, encoder):
        # TODO
        # Build a graph containing necessary operations and tensors
        # to train and predict multi-class data.

        # the labels will be np.array of shape (N, 1)
        # and np.int32 value in [0, n_classes)

        # the prediction should be based on the output of encoder.

        pass


class UnsupervisedTask(Task):
    def validate_data(self, data: UnsupervisedData):
        if len(data.shape) != 2:
            raise ValueError("Output data should be rank 2!")
        if data.shape[1] != self.output_dim:
            raise ValueError(
                f"Invalid output data dimension: {data.shape[1]} != {self.output_dim}!")


class AutoEncoderTask(UnsupervisedTask):

    def _extend_encode_graph(self, encoder):
        # TODO
        # Build a graph containing necessary operations and tensors
        # to reconstruct the original input data.

        # the prediction should be based on the output of encoder.

        pass
