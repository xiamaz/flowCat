"""Remove events on axis edges.
"""
from __future__ import annotations
from typing import Dict, Union

import pandas as pd

from flowcat.dataset import fcs
from flowcat.utils.intervals import create_interval, inner_interval, series_in_interval
from sklearn.base import BaseEstimator, TransformerMixin


class EdgeEventFilter(TransformerMixin, BaseEstimator):

    def __init__(self, channel_intervals: Dict[str, Union[tuple, pd.Interval]]):
        """Intervals are either (x.xx, y.yy) or (None, x.xx) etc, they are always strict smaller"""
        self._intervals = {
            channel: create_interval(interval)
            for channel, interval in channel_intervals.items()
        }

    def fit(self, *_):
        return self

    def transform(self, X: fcs.FCSData, y=None) -> fcs.FCSData:
        data = X.data
        ranges = X.ranges

        all_contained = None
        for colname, interval in self._intervals.items():
            newinterval = inner_interval(ranges[colname], interval)
            contained = series_in_interval(data[colname], newinterval)
            print(sum(contained))
            if all_contained is None:
                all_contained = contained
            else:
                all_contained &= contained
            ranges[colname] = newinterval
        X.data = data.loc[all_contained, :]
        X.ranges = ranges
        return X
