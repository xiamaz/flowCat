"""Create simple overviews over created experiments."""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from .file_utils import (
    load_experiments, load_predictions, load_metadata,
    add_avg_stats, add_prediction_info
)
from .overview import count_groups_filter, group_stats
from .plotting import experiments_plot, plot_avg_roc_curves, avg_stats_plot
from .pd_latex import df_to_table

CLUSTERING = "output/clustering"
CLASSIFICATION = "output/classification"


class Overview:
    """Create an overview of a clustering and classification projects."""

    def __init__(self):
        self._clustering = None
        self._classification = None

    @property
    def clustering(self):
        if self._clustering is None:
            self._clustering = load_experiments(CLUSTERING)
        return self._clustering

    @property
    def classification(self):
        if self._classification is None:
            self._classification = load_experiments(CLASSIFICATION)
            self._classification = add_avg_stats(self._classification)
        return self._classification

    def plot_clustering_experiments(
            self,
            path: str = "report",
            filename: str = "experiment_overview.png"
    ):
        """Plot experiment information using altair."""
        data = count_groups_filter(self.clustering)

        chart = experiments_plot(data)

        cpath = os.path.join(path, filename)
        chart.save(cpath)

    def write_classification_table(
            self,
            path: str = "report",
            filename: str = "result.tex"
    ):
        data = group_stats(self.classification).round(2)

        # remove uninteresting experiments
        data = data.loc[
            ~data.index.get_level_values(
                "name"
            ).str.contains("all_groups|more_merged")
        ]

        # pretty print count information
        data["count"] = data["count"].astype("int32").apply(str)
        table_string = df_to_table(data)
        tpath = os.path.join(path, filename)
        with open(tpath, "w") as tfile:
            tfile.write(table_string)

    def write(self, path):
        self.plot_clustering_experiments(path)
        self.write_classification_table(path)


class Prediction:
    """Create prediction analysis results from classification."""
    def __init__(self):
        self._experiments = None

    @property
    def experiments(self):
        if self._experiments is None:
            self._experiments = load_experiments(CLASSIFICATION)
        return self._experiments

    def plot_experiment(self, row, path):
        """Create plots for the given experiment."""

        pname = "{}_{}_{}".format(*row.name)

        plotpath = os.path.join(path, "predictions_{}".format(pname))
        somiter_data = load_predictions(row["predictions"])
        metadata = load_metadata(row["path"])

        # fig = plot_avg_roc_curves(somiter_data)
        # fig.savefig(plotpath+"_auc.png", dpi=200)

        # chart = avg_stats_plot(somiter_data)
        # chart.save(plotpath+"_stats.png")
        return pd.Series(
            name=row.name,
            data=[
                metadata["note"],
                ", ".join(metadata["group_names"]),
            ],
            index=[
                "note",
                "groups",
            ]
        )

    def plot_experiments(self, path):
        """Return statistics average over multiple iterations."""

        prediction_data = add_prediction_info(self.experiments)
        meta = prediction_data.apply(
            lambda x: self.plot_experiment(x, path), axis=1
        )
        return meta

    def write(self, path):
        metadata = self.plot_experiments(path)
        table_data = df_to_table(metadata, "llllp{6cm}")
        tpath = os.path.join(path, "prediction_meta.tex")
        with open(tpath, "w") as tfile:
            tfile.write(table_data)


def generate_report(path):
    os.makedirs(path, exist_ok=True)
    # Overview().write(path)
    Prediction().write(path)
