"""
Base classes to combine for each and combined transformators.
"""

from typing import List

import logging
from functools import wraps
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


LOGGER = logging.getLogger(__name__)


def add_history(fun):
    """Store function history in history dictionary."""
    @wraps(fun)
    def inner(self, cases, *_):
        self._history[fun.__name__] += [
            {
                "id": data.parent.id,
                "group": data.parent.group,
                "infiltration": data.parent.infiltration,
            }
            for data in cases
        ]
        return fun(self, cases, *_)
    return inner


class ApplySingle(BaseEstimator, TransformerMixin):
    """Wrap case information in the transformer."""

    def __init__(self, transformer):
        self._model = transformer

    def fit(self, X, *_):
        if self._model:
            self._model.fit(X)
        return self

    @property
    def history(self):
        if "out" in self._model.named_steps:
            return self._model.named_steps["out"].history
        return []

    def transform(self, X, *_):
        if self._model:
            X = self._model.transform(X)
        return X

    def predict(self, X, *_):
        return self._model.predict(X)


class Merge(BaseEstimator, TransformerMixin):
    """Merge different cases into a single dataframe for further analysis.
    Apply specific functions to each case before transforming and
    fitting the reduction model.

    Operate on a list of cases level.
    """

    def __init__(self, transformer, eachfit=None, eachtrans=None):
        self.model = transformer
        self.eachfit = ApplySingle(eachfit)
        self.eachtrans = ApplySingle(eachtrans)
        self._history = defaultdict(list)

    @property
    def history(self):
        return {
            "fit": self.eachfit.history if self.eachfit else [],
            "trans": self.eachtrans.history if self.eachtrans else [],
        }

    @add_history
    def fit(self, X: list, *_):
        """Fit model using a list a case paths."""
        fcs_data = [d.data for d in X]

        processed = map(self.eachfit.fit_transform, fcs_data)

        data = pd.concat(processed)
        self.model.fit(data)
        return self

    @add_history
    def transform(self, X: list, *_) -> list:
        """Transform a list of case paths into a list of case paths including
        the transformed result."""
        for data in X:
            LOGGER.info("%s:%s transform", data.parent.group, data.parent.id)
            data.result = self.model.transform(
                self.eachtrans.fit_transform(data.data)
            )
        return X
