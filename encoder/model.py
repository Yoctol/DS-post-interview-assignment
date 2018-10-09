import tensorflow as tf


class MultiTaskModel:
    def __init__(self, encoder):
        self.encoder = encoder
        self._task = set()
        self.graph = encoder.graph

    def add_task(self, task):
        if task in self._task:
            raise RuntimeError("Task already exists!")
        self._task.add(task)
        with tf.variable_scope(task.name), self.graph.as_default():
            task.build_graph()

    def fit(self, supervised, unsupervised):
        pass

    def evaluate(self):
        pass

    def export_encoder(self, path):
        self.encoder.save(path)
