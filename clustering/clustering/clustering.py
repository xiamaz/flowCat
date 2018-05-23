import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .tfsom.tfsom import SelfOrganizingMap
from .case_collection import CaseCollection


class FCSLogTransform(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, X, *_):
        names = [n for n in X.columns if "LIN" not in n]
        X[names] = np.log1p(X[names])
        return X

    def fit(self, *_):
        return self


class ClusteringTransform(BaseEstimator, TransformerMixin):
    def __init__(self, m, n, batch_size):
        self.m = m
        self.n = n
        self.batch_size = batch_size

    def fit(self, X, *_):
        # Build the TensorFlow dataset pipeline per the standard tutorial.
        self.graph = tf.Graph()
        with self.graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices(
                X.astype(np.float32)
            )
            num_inputs, dims = X.shape

            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            self.session = tf.Session(
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False
                )
            )
            self._model = SelfOrganizingMap(
                m=self.m, n=self.n, dim=dims, max_epochs=10, gpus=1,
                session=self.session, graph=self.graph, input_tensor=next_element,
                batch_size=self.batch_size, initial_learning_rate=0.05
            )
            init_op = tf.global_variables_initializer()
            self.session.run([init_op])
            self._model.train(num_inputs=num_inputs)

    def transform(self, X, *_):
        result = self._model.transform(X)
        return result / np.sum(result)


def create_pipeline(m=10, n=10, batch_size=4096):
    pipe = Pipeline(
        steps=[
            ("log", FCSLogTransform()),
            ("scale", StandardScaler()),
            ("clust", ClusteringTransform(m, n, batch_size)),
        ]
    )
    return pipe
