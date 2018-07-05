"""sklearn-like transform classes for case objects."""
import logging
from functools import wraps
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter


LOGGER = logging.getLogger(__name__)


def all_in(smaller, larger):
    for item in smaller:
        if item not in larger:
            return False
    return True


def filterstats(func):

    @wraps(func)
    def wrapper(self, X, *_):
        prev = sum(map(len, X.values()))

        result = func(self, X, *_)

        after = sum(map(len, result.values()))
        LOGGER.info("%s: Removed %d", self.__class__.__name__, prev - after)

        return result

    return wrapper


class SelectedMarkers:

    def __init__(self, threshold=0.9):
        self._threshold = threshold
        self._marker_counts = None
        self._marker_ratios = None
        self._selected_markers = None

    @property
    def selected_markers(self):
        return self._selected_markers

    def fit(self, X, *_):
        tube_markers = {}
        for case in X:
            for filepath in case["destpaths"]:
                tube_markers.setdefault(filepath["tube"], []).extend(
                    filepath["markers"]
                )
        self._marker_counts = {
            t: Counter(m) for t, m in tube_markers.items()
        }
        self._marker_ratios = {
            t: {k: c[k]/len(X) for k in c}
            for t, c in self._marker_counts.items()
        }
        self._selected_markers = {
            t: [v for v, r in c.items() if r >= self._threshold]
            for t, c in self._marker_ratios.items()
        }
        return self

    def predict(self, X, *_):
        predictions = [
            path
            for path in X["destpaths"]
            if all_in(self._selected_markers[path["tube"]], path["markers"])
        ]
        return predictions

    def transform(self, X, *_):
        X["destpaths"] = self.predict(X)
        return X


class SelectedTubes:

    def __init__(self, tubes=[1, 2], duplicate_allowed=False):
        self._tubes = tubes
        if duplicate_allowed:
            self._num_fun = lambda x: x >= 1
        else:
            self._num_fun = lambda x: x == 1

    def fit(self):
        return self

    def predict(self, X, *_):
        available_tubes = [int(p["tube"]) for p in X["destpaths"]]
        tube_counts = Counter(available_tubes)
        return all(map(lambda x: self._num_fun(tube_counts[x]), self._tubes))

    def transform(self, X, *_):
        return X if self.predict(X) else None

class MarkerFilter:

    def __init__(self, *args, **kwargs):
        self._selected = SelectedMarkers(*args, **kwargs)

    @property
    def selected_markers(self):
        return self._selected.selected_markers

    def fit(self, X, *_):
        cases = [c for cc in X.values() for c in cc]
        self._selected.fit(cases)
        return self

    @filterstats
    def transform(self, X, *_):
        filtered = {
            cohort: [
                case for case in
                [self._selected.transform(case) for case in cases]
                if case["destpaths"]
            ]
            for cohort, cases in X.items()
        }
        return filtered


class TubesFilter:

    def __init__(self, *args, **kwargs):
        self._model = SelectedTubes(*args, **kwargs)

    def fit(self, X, *_):
        return self

    @filterstats
    def transform(self, X, *_):
        filtered = {
            cohort: [
                case for case in
                [self._model.transform(case) for case in cases]
                if case
            ] for cohort, cases in X.items()
        }
        return filtered
