"""Remove events on axis edges.
"""
from __future__ import annotations
from typing import Dict, Union
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from flowcat.dataset import fcs

from . import FCSDataMixin


LOGGER = logging.getLogger(__name__)


class EdgeEventFilter(FCSDataMixin, TransformerMixin, BaseEstimator):

    def __init__(self, channels: list):
        """Intervals are either (x.xx, y.yy) or (None, x.xx) etc, they are always strict smaller"""
        self.trained = False
        self.channels = channels

    def fit(self, X: fcs.FCSData, *_):
        """Fit model to extreme values in channels for sample."""
        data = X.data
        channels = X.channels
        colindexes = [channels.index(c) for c in self.channels]
        sel_cols = data[:, colindexes]
        self._mins = sel_cols.min()
        self._maxs = sel_cols.max()
        self.trained = True
        return self

    def transform(self, X: fcs.FCSData, *_) -> fcs.FCSData:
        """Remove all events outside of specified intervals.
        """
        if not self.trained:
            raise RuntimeError("Model has not been trained yet.")

        data = X.data
        channels = X.channels

        sel_data = X.data[:, [channels.index(c) for c in self.channels]]
        all_contained = ((sel_data > self._mins) & (sel_data < self._maxs)).all(axis=1)

        LOGGER.info("Filtered out %s/%s", sum(all_contained), data.shape[0])
        X.data = data[all_contained, :]
        X.mask = data[all_contained, :]
        return X


class EmptyChannelFilter(FCSDataMixin, TransformerMixin, BaseEstimator):

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        return X.drop_empty()
