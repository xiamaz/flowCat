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

from .pd_latex import df_save_latex


SUBTABLE_TEMP = r"""\begin{{subtable}}{{.49\textwidth}}
  \centering
  \caption{{{}}}
  \resizebox{{\columnwidth}}{{!}}{{
    \input{{{}}}
  }}
\end{{subtable}}"""


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
            data = add_avg_stats(self.classification_files)
            data = self.extend_metadata(data)
            self._classification = data
        return self._classification

    @property
    def classification_by_groups(self):
        grouped = list(self.classification.groupby("groups"))
        for name, data in sorted(
                grouped, key=lambda x: len(x[0].split(", ")), reverse=True
        ):
            yield name, data

    def write_classification_table(
            self,
            path: "Path",
    ):
        output_folder = path / "classification_overview"
        output_folder.mkdir(exist_ok=True, parents=True)

        part_tables = {}
        for i, (name, data) in enumerate(self.classification_by_groups):
            data = group_stats(data).round(2)

            # remove uninteresting experiments
            data = data.loc[
                ~data.index.get_level_values(
                    "name"
                ).str.contains("all_groups|more_merged")
            ]

            # pretty print count information
            data["count"] = data["count"].astype("int32").apply(str)

            uniq = "".join(map(lambda x: x[0], name.split(", ")))
            # save data to latex table
            tpath = output_folder / "{}_{}.latex".format(i, uniq)
            data.to_latex(tpath)
            part_tables[name] = tpath.relative_to("figures")
            # df_save_latex(data, tpath)

        # generate combined figure
        parts = [
            SUBTABLE_TEMP.format(k, v)
            for k, v in part_tables.items()
        ]
        with open(str(path / "classification_overview.latex"), "w") as fobj:
            fobj.write("\n".join(parts))

    def show(self):
        """Printed output for the visualizer."""
        for name, group in self.classification.groupby("groups"):
            print(name)
            print(group)

    def write(self, path):
        """Written output such as tables and graphics for the visualizer."""
        self.write_classification_table(path)
