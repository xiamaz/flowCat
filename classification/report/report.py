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
from .predictions import df_get_predictions_t1
from .pd_latex import df_to_table

CLUSTERING = "output/clustering"
CLASSIFICATION = "output/classification"


def df_save_latex(data: pd.DataFrame, path: str, *args, **kwargs):
    table = df_to_table(data, *args, **kwargs)
    with open(path, "w") as tfile:
        tfile.write(table)


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

        # save data to latex table
        tpath = os.path.join(path, filename)
        df_save_latex(data, tpath)

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

        rocpath = plotpath+"_auc.png"
        if not os.path.exists(rocpath):
            fig = plot_avg_roc_curves(somiter_data)
            fig.savefig(rocpath, dpi=200)
        else:
            print("{} already exists. Not recreating".format(rocpath))

        chartpath = plotpath+"_stats.png"
        if not os.path.exists(chartpath):
            chart = avg_stats_plot(somiter_data)
            chart.save(chartpath)
        else:
            print("{} already exists. Not recreating".format(chartpath))

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
        # create plots for each experiment
        metadata = self.plot_experiments(path)
        # additionally save metadata as latex table
        tpath = os.path.join(path, "prediction_meta.tex")
        df_save_latex(metadata, tpath, "llllp{6cm}")


class Misclassifications:
    """Collect and transform the misclassifications into an easier to process
    information collection."""

    def __init__(self):
        self._data = None
        self._misclassifications = None

    @property
    def data(self):
        """Get the basic experiment information as a pandas dataframe."""
        if self._data is None:
            data = add_prediction_info(
                load_experiments(CLASSIFICATION)
            )
            self._data = data.loc[
                data["name"].str.contains("dedup") &
                ~data["name"].str.contains("more_merged") &
                ~data["name"].str.contains("all_groups")
            ]

        return self._data

    @property
    def misclassifications(self):
        """Get the high-frequency and high-certainty misclassifications."""
        if self._misclassifications is None:
            dfs = self.data.apply(
                self.get_misclassifications, axis=1
            )
            mis_df = pd.concat(
                dfs.tolist(),
                keys=self.data.index
            )
            self._misclassifications = mis_df

        return self._misclassifications

    def get_high_frequency_missed(
            self,
            frequency: float = 0.8,
            certainty: float = 0.8,
            method: str = "micro",
    ) -> pd.DataFrame:
        """Get misclassifications with high frequency in and across experiment
        sets."""
        all_nums = sum(self.data["predictions"].apply(len))
        counts = self.misclassifications.groupby("id").apply(
            lambda r: sum(r["count"]))
        counts.name = "all_count"

        macro_count = self.misclassifications.groupby("id").apply(
            lambda r: sum(r["rel_count"]) / self.data.shape[0]
        )
        macro_count.name = "macro"
        freq_df = pd.DataFrame([counts, macro_count]).T
        result = self.misclassifications.join(freq_df)

        result = result.mean(level=["id", "group", "prediction"])

        if method == "micro":
            freq_sel = result["all_count"] / all_nums >= frequency
        elif method == "macro":
            freq_sel = result["macro"].astype("float32") >= frequency
        else:
            raise RuntimeError("Unsupported averaging method.")

        result = result.loc[(result["mean"] >= certainty) & freq_sel]
        print(result)
        return result

    @staticmethod
    def get_misclassifications(data: pd.Series) -> pd.Series:
        """Get misclassifications for given slice of experiments with each
        misclassification tagged with direction and number of occurences"""
        predictions = load_predictions(data["predictions"])
        pdatas = {
            k: df_get_predictions_t1(v)
            for k, v in predictions.items()
        }
        pdata = pd.concat(pdatas.values(), keys=pdatas.keys())

        pdata.index.names = ["exp", "id"]
        all_misclassified = pdata.loc[pdata["group"] != pdata["prediction"]]
        all_misclassified.set_index(
            ["group", "prediction"], append=True, inplace=True
        )

        merged_misclassified = all_misclassified.groupby(
            level=["id", "group", "prediction"]
        ).apply(
            merge_misclassified
        )

        counts = all_misclassified.groupby(
            level=["id", "group", "prediction"]
        ).size()
        counts.name = "count"
        rel_counts = counts / len(data["predictions"])
        rel_counts.name = "rel_count"
        cdf = pd.DataFrame([counts, rel_counts]).T

        misclassif_result = merge_multi(
            merged_misclassified, cdf, ["id", "group", "prediction"]
        )

        return misclassif_result

    def write(self, path):
        tpath = os.path.join(path, "misclassifications.tex")
        csv_path = os.path.join(path, "misclassifications.csv")

        hi_freq = self.get_high_frequency_missed(method="macro")

        df_save_latex(hi_freq, tpath)
        hi_freq.to_csv(csv_path)


def merge_misclassified(data: pd.DataFrame) -> pd.DataFrame:
    """Merge misclassification information."""
    return pd.Series({
        "mean": float(data.mean()),
        "std": float(data.std()) if data.shape[0] > 1 else 0.0,
    })


def merge_multi(self, df, on):
    return self.reset_index().join(
        df, on=on
    ).set_index(self.index.names)


def generate_report(path):
    os.makedirs(path, exist_ok=True)
    # Overview().write(path)
    # Prediction().write(path)
    Misclassifications().write(path)
