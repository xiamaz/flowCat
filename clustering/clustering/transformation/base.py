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
from sklearn.pipeline import Pipeline

from .fcs import ScatterFilter, MarkersTransform
from .tfsom import SelfOrganizingMap, SOMNodes
from .pregating import SOMGatingFilter
from ..utils import save_json


LOGGER = logging.getLogger(__name__)


class PipelineBuilder():
    """Create a pipeline with the ability to store the build process
    for rebuilding."""

    # autogenerate string mapping of class names
    models = {
        m.__name__: m for m in [
            ScatterFilter, MarkersTransform, SelfOrganizingMap, SOMNodes,
            SOMGatingFilter
        ]
    }

    def __init__(self, name=""):
        """
        :param name: Name of the pipeline, mostly used for saving purposes.
        """
        # list of step parameters
        self.steps = []
        self.name = name

    def save(self, path: str):
        """
        :param path: Save the pipeline to the specified location.
        """
        save_json(os.path.join(path, self.name + ".json"), self.steps)

    def append(self, name, model, *args, **kwargs):
        self.steps.append({
            "name": name,
            "model": model,  # string model name for easier serialization
            "args": args,
            "kwargs": kwargs,
        })

    def build(self):
        return Pipeline(steps=[
            (
                s["name"], self.models[s["model"]](*s["args"], **s["kwargs"])
            ) for s in self.steps
        ])


def get_main(name: str, pipeline_name: str = "", *_) -> Pipeline:
    """Build main pipeline components."""
    builder = PipelineBuilder(name=pipeline_name)
    if name == "normal":
        builder.append(
            "clust",
            "SelfOrganizingMap",
            m=10, n=10
        )
    elif name == "gated":
        channels = ["CD45-KrOr", "SS INT LIN"]
        positions = ["+", "-"]
        # first gate merged results in fitting and individual
        # cases in transformation
        builder.append(
            "somgating",
            "SOMGatingFilter",
            channels,
            positions
        )
        builder.append(
            "somcluster",
            "SelfOrganizingMap",
            m=10,
            n=10,
        )
    else:
        raise RuntimeError("Unknown main transformation.")

    return builder


def get_pre(name: str, markers: list, pipeline_name: str = "", *_) -> Pipeline:
    """Build preprocessing components, which are applied to each case
    individually."""
    builder = PipelineBuilder(name=pipeline_name)
    # basic preprocessing used in all preprocessings
    builder.append(
        "scatter",
        "ScatterFilter",
    )
    if name == "normal":
        # default preprocessing
        pass
    elif name == "som":
        builder.append(
            "out",
            "SOMNodes",
            20, 20, 512
        )
    elif name == "gated":
        channels = ["CD45-KrOr", "SS INT LIN"]
        positions = ["+", "-"]
        builder.append(
            "out",
            "SOMGatingFilter",
            channels,
            positions,
        )
        # remove channels used in pregating from later usage
        markers = [m for m in markers if m not in channels]
    elif name == "gatedsom":
        channels = ["CD45-KrOr", "SS INT LIN"]
        positions = ["+", "-"]
        builder.append(
            "out",
            "SOMGatingFilter",
            channels,
            positions,
        )
        builder.append(
            "nodes",
            "SOMNodes",
            20, 20, 512
        )
        # remove channels used in pregating from later usage
        markers = [m for m in markers if m not in channels]
    else:
        raise RuntimeError("Unknown preprocesing")

    builder.append(
        "marker",
        "MarkersTransform",
        markers
    )
    return builder


def add_history(fun):
    """Store function history in history dictionary."""
    @wraps(fun)
    def inner(self, cases, *_):
        """Add metadata for each case into the history list.  This can be later
        zipped with other model internal information to create rich
        representations of internal processes."""
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

    def __init__(
            self, main: Pipeline, fit: Pipeline, trans: Pipeline, steps: dict,
    ):
        self.model = main
        self.eachfit = fit
        self.eachtrans = trans
        self.steps = steps
        self._history = defaultdict(list)

    @classmethod
    def from_names(cls, main: str, prefit: str, pretrans: str, markers: list):
        """Choose pipeline components from provided names."""
        builders = {
            "main": get_main(main, "main"),
            "fit": get_pre(prefit, markers, "fit"),
            "trans": get_pre(pretrans, markers, "trans"),
        }
        return cls(
            **{k: v.build() for k, v in builders.items()},
            steps=builders
        )

    @classmethod
    def load(cls, path: str):
        """Load existing model from the given folder."""
        # create the steps for the pipeline
        # somehow load the state required for the individual steps
        pass

    @property
    def history(self):
        return {
            "fit": self.eachfit.history if self.eachfit else [],
            "trans": self.eachtrans.history if self.eachtrans else [],
        }

    def save(self, path: str):
        """Save the current model into the specified directory."""
        # save the steps necessary to create the models
        self.steps.save(path)

        # save stateful and stateless models
        # for stateful steps save the trained weights in an individual manner

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
