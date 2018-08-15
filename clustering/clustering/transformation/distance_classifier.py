from collections import defaultdict

import pandas as pd

from .tfsom import TFSom, SOMNodes
from sklearn.base import TransformerMixin, BaseEstimator


class DistanceClassifier(TransformerMixin, BaseEstimator):
    """Classificatin based on distance of nodes in SOM to the current
    provided events."""

    def __init__(self):
        self._models = {}

        self._premodel = SOMNodes()

    def fit(self, X, *_):
        """Input a list of TubeCase objects, to be fitted to individual
        SOM models."""

        groups = defaultdict(list)
        for tubecase in X.data:
            groups[tubecase.parent.group].append(tubecase)
        for name, cases in groups.items():
            print("Training {}".format(name))
            self._models[name] = TFSom(
                m=10,
                n=10,
                max_epochs=10,
                softmax_activity=False
            )
            casedata = pd.concat([
                self._premodel.fit_transform(c.data[X.markers]) for c in cases
                if c.data.shape[0] > 10000
            ])
            self._models[name].train(casedata)

        return self

    def _calculate_distances(self, X, *_):
        """Calculate the distance of the given case to all contained models."""
        predictions = {}
        for name, model in self._models.items():
            predictions[name] = model.distance_to_map(X)
        return predictions

    def predict(self, X, *_):
        """Get the predicted class for the given TubeView."""
        pred_records = []
        for tubecase in X.data:
            if tubecase.data.shape[0] <= 1000:
                tubecase.result_success = False
                tubecase.result = "{} <= 1000 required".format(
                    tubecase.data.shape[0]
                )
                continue

            distances = self._calculate_distances(
                self._premodel.fit_transform(tubecase.data[X.markers])
            )
            prediction = min(distances, key=distances.get)

            tubecase.result = distances
            tubecase.result_success = True

            pred_records.append(
                {**{"prediction": prediction}, **tubecase.metainfo_dict}
            )

        return pd.DataFrame.from_records(pred_records)
