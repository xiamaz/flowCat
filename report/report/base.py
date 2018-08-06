"""Base reporting objects."""
from pathlib import Path
from enum import Enum

import pandas as pd

from .file_utils import load_experiments, load_metadata


class ExpType(Enum):
    """Type of experiment."""
    CLUST = 1
    CLASS = 2


class SplitType(Enum):
    """Type of experiment split."""
    KFOLD = 1
    HOLDOUT = 2


class Reporter:
    def __init__(
            self,
            path: "Path" = "../output",
    ):
        self._basepath = Path(path)
        self._path_classification = self._basepath / "classification"
        self._path_clustering = self._basepath / "clustering"

        self._clustering_files = None
        self._classification_files = None

    @property
    def clustering_files(self):
        if self._clustering_files is None:
            data = load_experiments(self._path_clustering)
            data["exptype"] = ExpType.CLUST
            self._clustering_files = data
        return self._clustering_files

    @property
    def classification_files(self):
        if self._classification_files is None:
            data = load_experiments(
                self._path_classification
            )
            data["exptype"] = ExpType.CLASS
            self._classification_files = data
        return self._classification_files

    @staticmethod
    def get_metadata(experiment_row):
        if experiment_row["exptype"] != ExpType.CLASS:
            raise TypeError("Experiment is not a classification experiment")
        return load_metadata(experiment_row["path"])

    @staticmethod
    def get_merged_metadata(experiment_row):
        if experiment_row["exptype"] != ExpType.CLASS:
            raise TypeError("Experiment is not a classification experiment")
        metadata = load_metadata(experiment_row["path"])
        # take the first split info only, since we are only summing up the
        # values
        splits = [
            exp["splits"][0]
            for meta in metadata
            for exp in meta["experiments"]
        ]
        group_sizes = {
            group: splits[0][
                "test"
            ]["groups"][group] + splits[0][
                "train"
            ]["groups"][group]
            for group in splits[0]["test"]["groups"]
        }
        max_size = max(group_sizes.values())
        min_size = min(group_sizes.values())

        return pd.Series(
            {
                "groups": ", ".join(group_sizes.keys()),
                "num": len(splits),
                "max_size": max_size,
                "min_size": min_size
            }
        )

    @classmethod
    def extend_metadata(cls, experiments):
        experiments[
            ["groups", "num", "max_size", "min_size"]
        ] = experiments.apply(
            cls.get_merged_metadata, axis=1
        )
        return experiments

    def get_experiment_sets(self, experiments):
        """Get experiment groups, that should be processed together."""
        exp_sets = list(experiments.groupby(by=["set", "name", "type"]))
        return exp_sets
