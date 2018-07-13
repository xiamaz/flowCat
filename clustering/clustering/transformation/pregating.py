import math
import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN

from .tfsom import SOMNodes


LOGGER = logging.getLogger(__name__)


class GatingFilter(BaseEstimator, TransformerMixin):
    """Specify axis and either negative or positive for each to select
    a cell population located there.

    The algorithm is density based, thus requires adjustments to work with
    different data densities.

    Close clusters are merged for the result, this should capture multi-center
    cell populations.
    """

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
                (xref - means[xchan])**2 + (yref - means[ychan])**2
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
                (xref - means[xchan])**2 + (yref - means[ychan])**2
            )
            if abs(cdist - dist) < self._merge_dist:
                LOGGER.debug(
                    "Merging %d because dist diff %d",
                    cl_num,
                    abs(cdist - dist)
                )
                merged.append(cl_num)

        sel_res = np.zeros(predictions.shape)
        for sel_cluster in merged:
            sel_res[predictions == sel_cluster] = 1

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
        self._model = DBSCAN(
            eps=self._eps, min_samples=min_samples, metric="manhattan"
        )
        return self


class SOMGatingFilter(BaseEstimator, TransformerMixin):
    """Create SOM for individual files and thereafter apply pregating to
    each case."""

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
