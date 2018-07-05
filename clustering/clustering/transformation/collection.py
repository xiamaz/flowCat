from collections import defaultdict
import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


LOGGER = logging.getLogger(__name__)


class ApplySingle(BaseEstimator, TransformerMixin):
    """Wrap case information in the transformer."""

    def __init__(self, transformer):
        self._model = transformer
        self._history = []

    @property
    def history(self):
        if "out" in self._model.named_steps:
            return [
                {**h, **m} for h, m in
                zip(self._model.named_steps["out"].history, self._history)
            ]
        else:
            return []

    def fit(self, X, *_):
        meta, data = X
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        meta, data = X
        tdata = self._model.transform(data)

        # add metainformation to history for tracking
        label, cohort, infil = meta
        self._history.append({
            "text": "{} {} I{}".format(label[1], cohort[1], infil[1])
        })

        return meta, tdata

    def predict(self, X, *_):
        meta, data = X
        pdata = self._model.predict(data)
        return meta, pdata


class Merge(BaseEstimator, TransformerMixin):
    """Merge different cases into a single dataframe for further analysis.
    Apply specific functions to each case before transforming and
    fitting the reduction model."""

    def __init__(self, transformer, eachfit=None, eachtrans=None):
        self.model = transformer
        self.eachfit = ApplySingle(eachfit) if eachfit else None
        self.eachtrans = ApplySingle(eachtrans) if eachtrans else None

    @property
    def history(self):
        return {
            "fit": self.eachfit.history if self.eachfit else [],
            "trans": self.eachtrans.history if self.eachtrans else [],
        }

    def fit(self, X, *_):
        if self.eachfit:
            processed = [self.eachfit.fit_transform(d) for d in X]
        else:
            processed = X
        data = pd.concat([d for _, d in processed])
        self.model.fit(data)
        return self

    def transform(self, X, *_):
        meta_dict = defaultdict(list)
        results = []
        for (meta, data) in X:
            if self.eachtrans:
                meta, data = self.eachtrans.fit_transform((meta, data))

            tdata = self.model.transform(data)
            results.append(tdata)

            for (key, item) in meta:
                meta_dict[key].append(item)

        return pd.concat(
            [pd.DataFrame(np.matrix(results)), pd.DataFrame(meta_dict)],
            axis=1
        )
