import math

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import cluster
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


class ScatterFilter(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            filters=[("SS INT LIN", 0), ("FS INT LIN",)],
    ):
        self._filters = filters

    def transform(self, X, *_):
        for column, threshold in self._filters:
            X = X[X[column] > threshold]
        return X

    def fit(self, *_):
        return self


class GatingFilter(BaseEstimator, TransformerMixin):

    def __init__(self, channels, positions):
        self._channels = channels
        self._positions = positions
        self._model = None

    def _select_position(self, X, predictions):
        (xchan, ychan) = self._channels
        (xpos, ypos) = self._channels
        xref = 1023 if xpos == "+" else 0
        yref = 1023 if ypos == "-" else 0
        closest = None
        cdist = None
        for cl_num in np.unique(predictions):
            if cl_num == -1:
                continue
            means = X.loc[predictions == cl_num, self._channels].mean(
                axis=0
            )
            dist = math.sqrt(
                (xref-means[xchan])**2 + (yref-means[ychan])**2
            )
            if closest is None or cdist > dist:
                closest, cdist = cl_num, dist

        sel_res = np.zeros(predictions.shape)
        sel_res[predictions == closest] = 1
        return sel_res

    def transform(self, X, *_):
        predictions = self._model.fit_predict(X[self._channels].values)
        selected = self._select_position(X, predictions)
        return X[selected]

    def fit(self, X, *_):
        sample_num = X.shape[0]
        min_samples = 300 * (sample_num / 50000)
        self._model = cluster.DBSCAN(
            eps=30, min_samples=min_samples, metric="manhattan"
        )
        return self


class ClusteringTransform(BaseEstimator, TransformerMixin):
    def __init__(self, m, n, batch_size):
        self.m = m
        self.n = n
        self.batch_size = batch_size
        self.graph = None
        self.model = None

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
            self.model = SelfOrganizingMap(
                m=self.m, n=self.n, dim=dims, max_epochs=10, gpus=1,
                session=self.session, graph=self.graph, input_tensor=next_element,
                batch_size=self.batch_size, initial_learning_rate=0.05
            )
            init_op = tf.global_variables_initializer()
            self.session.run([init_op])
            self.model.train(num_inputs=num_inputs)

    def transform(self, X, *_):
        result = self.model.transform(X)
        return result / np.sum(result)

    def predict(self, X, *_):
        result = self.model.predict(X)
        return result


def create_pipeline(m=10, n=10, batch_size=4096):
    pipe = Pipeline(
        steps=[
            # ("log", FCSLogTransform()),
            ("scale", StandardScaler()),
            ("clust", ClusteringTransform(m, n, batch_size)),
        ]
    )
    return pipe
