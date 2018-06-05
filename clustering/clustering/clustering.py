import math
import logging
from collections import Counter

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from .tfsom.tfsom import SelfOrganizingMap
from .case_collection import CaseCollection
from .plotting import plot_overview


LOGGER = logging.getLogger(__name__)


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
            filters=[("SS INT LIN", 0), ("FS INT LIN", 0)],
    ):
        self._filters = filters

    def transform(self, X, *_):
        for column, threshold in self._filters:
            X = X[X[column] > threshold]
        return X

    def fit(self, *_):
        return self


class GatingFilter(BaseEstimator, TransformerMixin):

    def __init__(
            self, channels, positions, min_samples=None, eps=30, merge_dist=200
    ):
        self._channels = channels
        self._positions = positions
        self._model = None
        self._min_samples = min_samples
        self._eps = eps
        self._merge_dist = merge_dist

    def _select_position(self, X, predictions):
        (xchan, ychan) = self._channels
        (xpos, ypos) = self._positions
        xref = 1023 if xpos == "+" else 0
        yref = 1023 if ypos == "+" else 0
        closest = None
        cdist = None
        for cl_num in np.unique(predictions):
            if cl_num == -1:
                continue
            means = X.loc[predictions == cl_num, self._channels].mean(axis=0)
            dist = math.sqrt(
                (xref-means[xchan])**2 + (yref-means[ychan])**2
            )
            if closest is None or cdist > dist:
                closest, cdist = cl_num, dist

        # merge clusters close to the closest cluster
        merged = [closest]
        for cl_num in np.unique(predictions):
            # skip background and selected cluster
            if cl_num == -1 or cl_num == closest:
                continue
            means = X.loc[predictions == cl_num, self._channels].mean(axis=0)
            dist = math.sqrt(
                (xref-means[xchan])**2 + (yref-means[ychan])**2
            )
            if abs(cdist - dist) < self._merge_dist:
                LOGGER.debug(
                    "Merging %d because dist diff %d", cl_num, abs(cdist- dist)
                )
                merged.append(cl_num)

        sel_res = np.zeros(predictions.shape)
        for cluster in merged:
            sel_res[predictions == cluster] = 1

        return sel_res

    def transform(self, X, *_):
        selected = self.predict(X)
        return X[selected == 0]

    def predict(self, X, *_):
        predictions = self._model.fit_predict(X[self._channels].values)
        # select clusters based on channel position
        selected = self._select_position(X, predictions)
        # return 1/0 array based on inclusion in gate or not
        return selected

    def fit(self, X, *_):
        sample_num = X.shape[0]
        if self._min_samples:
            min_samples = self._min_samples
        else:
            min_samples = 300 * (sample_num / 50000)
        self._model = cluster.DBSCAN(
            eps=self._eps, min_samples=min_samples, metric="manhattan"
        )
        return self


class ClusteringTransform(BaseEstimator, TransformerMixin):
    def __init__(self, m=10, n=10, batch_size=4096, test_batch_size=8192):
        self.m = m
        self.n = n
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
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
                session=self.session, graph=self.graph,
                input_tensor=next_element, batch_size=self.batch_size,
                test_batch_size=self.test_batch_size,
                initial_learning_rate=0.05
            )
            init_op = tf.global_variables_initializer()
            self.session.run([init_op])
            self.model.train(num_inputs=num_inputs)
        return self

    def transform(self, X, *_):
        """Return relative distribution of all events in the created SOM."""
        result = self.model.transform(X)
        return result / np.sum(result)

    def predict(self, X, *_):
        """Map all events to their respective nodes as a list of
        event-length."""
        result = self.model.predict(X)
        return result


class SOMNodes(BaseEstimator, TransformerMixin):

    def __init__(self, m=10, n=10, batch_size=1024):
        self._model = ClusteringTransform(m, n, batch_size=batch_size)
        self.history = []

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X, *_):
        return self._model.predict(X, *_)

    def transform(self, X, *_):
        self._model.fit(X)
        weights = pd.DataFrame(
            self._model.model.output_weights, columns=X.columns
        )
        self.history.append({
            "data": weights,
            "mod": weights.index,
        })
        return weights


class SOMGatingFilter(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            channels=["CD45-KrOr", "SS INT LIN"],
            positions=["+", "-"],
    ):
        # self._pre = ClusteringTransform(10, 10, 2048)
        self._pre = SOMNodes(10, 10, 2048)
        self._clust = GatingFilter(channels, positions, min_samples=4, eps=50)

        self._channels = channels

        self.history = []

    def fit(self, X, *_):
        # always fit new when predicting data
        return self

    def predict(self, X, *_):
        new_weights = self._pre.transform(X)
        self._clust.fit(new_weights)

        event_to_node = self._pre.predict(X)

        # get som nodes that are associated with clusters
        som_to_clust = self._clust.predict(new_weights)
        event_filter = np.vectorize(lambda x: som_to_clust[x])(event_to_node)

        # add to record lists for later plotting usage
        self.history.append({
            "data": new_weights,
            "mod": som_to_clust,
        })

        return event_filter

    def transform(self, X, *_):
        selection = self.predict(X, *_)
        result = X.loc[selection == 1, ~X.columns.isin(self._channels)]
        return result


def create_pipeline(m=10, n=10, batch_size=4096):
    pipe = Pipeline(
        steps=[
            ("clust", ClusteringTransform(m, n, batch_size)),
        ]
    )
    return pipe


def create_pipeline_multistage(
        channels=["CD45-KrOr", "SS INT LIN"],
        positions=["+", "-"],
        m=10, n=10
):
    pipe = Pipeline(steps=[
        ("somgating", SOMGatingFilter(channels, positions)),
        ("somcluster", ClusteringTransform(m=m, n=n)),
    ])
    return pipe


class MultiSOM(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            first=(20, 20, 512),
            second=(10, 10, 16),
    ):
        self._first = first
        self._second = second

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        return X


def create_presom_each(
        first=(20, 20, 512)
):
    pipe = Pipeline(steps=[
        ("scatter", ScatterFilter()),
        ("out", SOMNodes(*first)),
    ])
    return pipe

def create_pre(
):
    pipe = Pipeline(steps=[
        ("scatter", ScatterFilter()),
    ])
    return pipe

def create_pregate_nodes(
        channels=["CD45-KrOr", "SS INT LIN"],
        positions=["+", "-"],
        first=(20, 20, 512)
):
    pipe = Pipeline(steps=[
        ("scatter", ScatterFilter()),
        ("out", SOMGatingFilter(channels, positions)),
        ("nodes", SOMNodes(*first))
    ])
    return pipe

def create_pregate(
        channels=["CD45-KrOr", "SS INT LIN"],
        positions=["+", "-"],
):
    pipe = Pipeline(steps=[
        ("scatter", ScatterFilter()),
        ("out", SOMGatingFilter(channels, positions)),
    ])
    return pipe
