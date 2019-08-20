"""Basic transformation utilities for fcs."""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class FCSLogTransform(TransformerMixin, BaseEstimator):
    """Transform FCS files logarithmically.  Currently this does not work
    correctly, since FCS files are not $PnE transformed on import"""

    def transform(self, X, *_):
        names = [n for n in x.columns if "lin" not in n]
        X[names] = np.log1p(X[names])
        return X

    def fit(self, *_):
        return self
