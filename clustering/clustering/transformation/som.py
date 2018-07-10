"""
SOM scikit classes for transformations.

This class wraps the tensorflow self-organizing map, providing checks
for marker channels and handles save/restore functionality for these
additional metadata.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .tfsom import SelfOrganizingMap


class SOMNodes(BaseEstimator, TransformerMixin):
    """
    Create SOM from input data and transform into the weights
    for each SOM-Node, effectively reducing the data to num_nodes x channels
    dimensions.
    """

    def __init__(self, m=10, n=10, batch_size=1024):
        self._model = SelfOrganizingMap(m, n, batch_size=batch_size)
        self.history = []

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X, *_):
        return self._model.predict(X, *_)

    def transform(self, X, *_):
        self._model.fit(X)
        weights = pd.DataFrame(
            self._model.output_weights, columns=X.columns
        )
        self.history.append({
            "data": weights,
            "mod": weights.index,
        })
        return weights
