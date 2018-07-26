import os

import pandas as pd
from .file_utils import (
    load_experiments, load_predictions, load_metadata,
    add_avg_stats, add_prediction_info
)

from .prediction import df_get_predictions_t1

from .base import Reporter


def merge_multi(self, df, on):
    return self.reset_index().join(
        df, on=on
    ).set_index(self.index.names)


def get_infiltrations(data: pd.Series) -> pd.Series:
    """Get infiltration information on each id."""
    predictions = load_predictions(data["predictions"])
    infiltrations = pd.concat([
        p["infiltration"] for p in predictions.values()
    ])
    return infiltrations[~infiltrations.index.duplicated(keep='first')]


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

    print(merged_misclassified.columns)

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


def merge_misclassified(data: pd.DataFrame) -> pd.DataFrame:
    """Merge misclassification information."""
    return pd.Series({
        "mean": float(data.mean()),
        "std": float(data.std()) if data.shape[0] > 1 else 0.0,
    })


class Misclassifications(Reporter):
    """Collect and transform the misclassifications into an easier to process
    information collection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = None
        self._misclassifications_rel = None

    @property
    def data(self):
        """Get the basic experiment information as a pandas dataframe."""
        if self._data is None:
            data = add_prediction_info(
                self.classification_files
            )
            self._data = data

        return self._data

    def get_misclassifications(self, data):
        """Get the high-frequency and high-certainty misclassifications in
        the specified datasets.
        """
        dfs = data.apply(
            get_misclassifications, axis=1
        )
        mis_df = pd.concat(
            dfs.tolist(),
            keys=self.data.index
        )
        return mis_df

    def get_infiltrations(self, data):
        infils = []
        for _, row in data.iterrows():
            infils.append(get_infiltrations(row))

        infiltrations = pd.concat(infils)
        return infiltrations[~infiltrations.index.duplicated(keep='first')]

    @property
    def misclassifications_rel(self):
        """Misclassification table with relative counts."""

        if self._misclassifications_rel is None:
            # average by taking all individual numbers and dividing by the
            # overall total number
            all_nums = sum(self.data["predictions"].apply(len))
            counts = self.misclassifications.groupby("id").apply(
                lambda r: sum(r["count"])) / all_nums
            counts.name = "micro"

            # average by averaging individual total numbers, this will give
            # each group equal weight, but ignore size imbalances between
            # groups
            macro_count = self.misclassifications.groupby("id").apply(
                lambda r: sum(r["rel_count"]) / self.data.shape[0]
            )
            macro_count.name = "macro"

            # join calculations into the original dataframe
            freq_df = pd.DataFrame([counts, macro_count]).T
            result = self.misclassifications.join(freq_df)

            # cast results into the correct datatypes
            result["micro"] = result["micro"].astype("float32")
            result["macro"] = result["macro"].astype("float32")

            self._misclassifications_rel = result

        return self._misclassifications_rel

    @property
    def avg_missed(self):
        """Average over sets of experiments. This will not produce
        reasonable results for all columns.
        """
        set_averages = self.misclassifications_rel.mean(
            level=["id", "group", "prediction"]
        )
        return set_averages

    def get_high_frequency_missed(
            self,
            frequency: float = 0.8,
            certainty: float = 0.8,
            method: str = "micro",
    ) -> pd.DataFrame:
        """Get misclassifications with high frequency in and across experiment
        sets."""
        set_averages = self.avg_missed

        # filter frequency on the selected method
        freq_sel = set_averages[method].astype("float32") >= frequency

        # filter misclassification dataframe on both
        result = set_averages.loc[
            (set_averages["mean"] >= certainty) & freq_sel
        ]
        return result

    def write(self, path):
        tpath = os.path.join(path, "misclassifications.tex")
        csv_path = os.path.join(path, "misclassifications.csv")

        hi_freq = self.get_high_frequency_missed(
            frequency=1.0,
            method="macro"
        )
        print(hi_freq)

        df_save_latex(hi_freq, tpath)
        hi_freq.to_csv(csv_path)

        freq_plot_path = os.path.join(path, "missed_frequency.png")
        plot_frequency(self.avg_missed, freq_plot_path)
