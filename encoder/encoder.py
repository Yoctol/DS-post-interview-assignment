import tensorflow as tf

_SCOPE_NAME = "Encoder"


class Encoder:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.variable_scope(_SCOPE_NAME):
            self.build_graph()
        self.sess = tf.Session(graph=self.graph)
        self.saver = tf.train.Saver(
            var_list=self.graph.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=_SCOPE_NAME),
        )

    def build_graph(self):
        # TODO
        pass

    def encode(self, X):
        # TODO
        pass

    @classmethod
    def load(self, path):
        # TODO
        pass

    def save(self, path):
        self.saver.save(self.sess, path)
