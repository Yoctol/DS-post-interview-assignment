import numpy as np
import tensorflow as tf

from .types import Data

_SCOPE_NAME = "Encoder"


class Encoder:
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._set_up()

    def _set_up(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.variable_scope(_SCOPE_NAME):
            self._build_graph()
        self.sess = tf.Session(graph=self.graph)
        self.saver = tf.train.Saver(
            var_list=self.graph.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=_SCOPE_NAME),
        )
        self.sess.run(
            tf.variables_initializer(
                var_list=self.graph.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=_SCOPE_NAME,
                ),
            )
        )

    def _build_graph(self):
        # TODO
        pass

    def encode(self, X: np.ndarray) -> np.ndarray:
        self.validate_data(X)
        # TODO

    def validate_data(self, data: Data):
        x = data[0] if isinstance(data, tuple) else data
        if len(x.shape) != 2:
            raise ValueError("Input data should be rank 2!")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Invalid input data dimension: {x.shape[1]} != {self.input_dim}!")

    @classmethod
    def load(cls, path: str):
        # TODO
        pass

    def save(self, path: str):
        # TODO
        # save hyper parameters as well (if they exist.)
        pass
