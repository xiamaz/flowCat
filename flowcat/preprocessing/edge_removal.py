"""Remove events on axis edges.
"""
from __future__ import annotations
from typing import Dict, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from flowcat.dataset import fcs
from flowcat.utils.intervals import create_interval, inner_interval, series_in_interval

from . import FCSDataMixin


class EdgeEventFilter(FCSDataMixin, TransformerMixin, BaseEstimator):

    def __init__(self, channel_intervals: Dict[str, Union[tuple, pd.Interval]]):
        """Intervals are either (x.xx, y.yy) or (None, x.xx) etc, they are always strict smaller"""
        self._intervals = {
            channel: create_interval(interval)
            for channel, interval in channel_intervals.items()
        }

    def fit(self, *_):
        return self

    def transform(self, X: fcs.FCSData, y=None) -> fcs.FCSData:
        """Remove all events outside of specified intervals."""
        data = X.data
        meta = X.meta

        all_contained = None
        new_ranges = {}
        for colname, interval in self._intervals.items():
            newinterval = inner_interval(meta[colname].range, interval)
            contained = series_in_interval(data[colname], newinterval)

            if all_contained is None:
                all_contained = contained
            else:
                all_contained &= contained

            new_ranges[colname] = newinterval
        X.data = data.loc[all_contained, :]
        X.set_ranges(new_ranges)
        return X


class EmptyChannelFilter(FCSDataMixin, TransformerMixin, BaseEstimator):

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        return X.drop_empty()
