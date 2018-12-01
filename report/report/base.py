"""Base reporting objects."""
import sys
from pathlib import Path
from enum import Enum
from argparse import ArgumentParser
from contextlib import contextmanager

import pandas as pd

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from .file_utils import load_experiments, load_metadata


GROUP_NAME_MAP = {
    "CLLPL": "PL",
    "HZL": "HCL",
    "HZLv": "HCLv",
    "Mantel": "MCL",
    "Marginal": "MZL",
}


class ExpType(Enum):
    """Type of experiment."""
    CLUST = 1
    CLASS = 2


class SplitType(Enum):
    """Type of experiment split."""
    KFOLD = 1
    HOLDOUT = 2


@contextmanager
def plot_figure(plotpath, *args, **kwargs):
    """Provide context for plotting with automatic drawing and saving."""
    fig = Figure(*args, **kwargs)
    ax = fig.add_subplot(111)
    yield ax
    FigureCanvas(fig)
    fig.savefig(plotpath)


def create_parser(description=""):
    """Create basic parser defining input and output folder."""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--indir",
        help="Folder containing input data.",
        default="../output",
        type=Path
    )

    parser.add_argument(
        "--outdir",
        help="Output directory for plots and tables.",
        default="figures",
        type=Path
    )
    return parser


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

        groups = sorted([
            GROUP_NAME_MAP.get(g, g) for g in group_sizes.keys()
        ])

        return pd.Series(
            {
                "groups": ", ".join(groups),
                "grouplist": groups,
                "num": len(splits),
                "max_size": max_size,
                "min_size": min_size
            }
        )

    @classmethod
    def extend_metadata(cls, experiments):
        experiments[
            ["groups", "grouplist", "num", "max_size", "min_size"]
        ] = experiments.apply(
            cls.get_merged_metadata, axis=1
        )
        return experiments

    def get_experiment_sets(self, experiments):
        """Get experiment groups, that should be processed together."""
        exp_sets = list(experiments.groupby(by=["set", "name", "type"]))
        return exp_sets
