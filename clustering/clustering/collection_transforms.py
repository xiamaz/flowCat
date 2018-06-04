from collections import defaultdict
import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


LOGGER = logging.getLogger(__name__)


class ApplySingle(BaseEstimator, TransformerMixin):

    def __init__(self, transformer):
        self._model = transformer

    def fit(self, X, *_):
        meta, data = X
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        meta, data = X
        tdata = self._model.transform(data)
        return meta, tdata

    def predict(self, X, *_):
        meta, data = X
        pdata = self._model.predict(data)
        return meta, pdata


class Merge(BaseEstimator, TransformerMixin):

    def __init__(self, transformer, eachfit=None, eachtrans=None):
        self._model = transformer
        self._eachfit = ApplySingle(eachfit) if eachfit else None
        self._eachtrans = ApplySingle(eachtrans) if eachtrans else None

    def fit(self, X, *_):
        if self._eachfit:
            processed = [self._eachfit.fit_transform(d) for d in X]
        else:
            processed = X
        data = pd.concat([d for _, d in processed])
        self._model.fit(data)
        return self

    def transform(self, X, *_):
        meta_dict = defaultdict(list)
        results = []
        for (meta, data) in X:
            if self._eachtrans:
                meta, data = self._eachtrans.fit_transform((meta, data))

            tdata = self._model.transform(data)
            results.append(tdata)

            for (key, item) in meta:
                meta_dict[key].append(item)

        return pd.concat(
            [pd.DataFrame(np.matrix(results)), pd.DataFrame(meta_dict)],
            axis=1
        )
