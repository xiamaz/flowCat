"""
Transform operations for overview generation.
"""
from functools import reduce
from math import sqrt
import pandas as pd

from .file_utils import (
    load_experiments, load_predictions, load_metadata,
    add_avg_stats, add_prediction_info
)
from .plotting import (
    experiments_plot,
    plot_avg_roc_curves,
    avg_stats_plot,
    plot_frequency
)

from .base import Reporter


def group_avg_stat(data: pd.DataFrame) -> pd.Series:
    """Average statistics numbers in a single group."""
    count = data.shape[0]

    f1_score, var = tuple(map(
        lambda x: x / count,
        reduce(
            lambda ab, xy: (ab[0] + xy[0], ab[1] + xy[1]), data["f1"], (0, 0)
        )
    ))
    std = sqrt(var)

    return pd.Series(data=[
        count, f1_score, std
    ], index=[
        "count", "f1", "std"
    ])


def group_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Average stats across replications of the same experiment (with different
    date tags).
    """
    resp = data.groupby(["set", "name", "type"]).apply(group_avg_stat)
    return resp


def count_groups_filter(
        data: pd.DataFrame,
        threshold: int = 1
) -> pd.DataFrame:
    """Create group counts and return after filtering."""
    counts = pd.DataFrame(
        data.groupby(["set", "name", "type"]).size(), columns=["count"]
    )
    counts = counts.loc[counts["count"] > threshold]
    return counts


class Overview(Reporter):
    """Create an overview of a clustering and classification projects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._classification = None

    @property
    def classification(self):
        if self._classification is None:
            self.classification_files
            self._classification = add_avg_stats(self._classification)
        return self._classification

    def plot_clustering_experiments(
            self,
            path: str = "report",
            filename: str = "experiment_overview.png"
    ):
        """Plot experiment information using altair."""
        data = count_groups_filter(self.clustering_files)

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

        # save data to latex table
        tpath = os.path.join(path, filename)
        df_save_latex(data, tpath)

    def write(self, path):
        self.plot_clustering_experiments(path)
        self.write_classification_table(path)
