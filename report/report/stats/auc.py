"""
Functions for calculation of receiver operator characteristic and AUC measures
thereof."""
from functools import reduce

import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve

from ..file_utils import df_prediction_cols


def avg_roc_plot(somiter_data: dict, axes: "Axes") -> "Figure":
    """Create a roc plot using roc data per cohort in a dict
    and additional metadata (avg AUC) in auc dict."""
    roc_data = _prediction_dict_to_roc_df(somiter_data)
    group_auc = auc_per_group(somiter_data)

    colors = cm.tab20.colors  # pylint: disable=no-member
    for i, (name, data) in enumerate(roc_data.groupby(level=0)):
        axes.plot(
            data["fpr"],
            data["tpr"],
            color=colors[i * 2], lw=1,
            label="{} (AUC {:.2f})".format(name, group_auc[name])
        )
        axes.fill_between(
            data["fpr"],
            data["tpr"] - data["std"],
            data["tpr"] + data["std"],
            color=colors[i * 2 + 1],
            alpha=1 - (1 / len(group_auc) * i),
        )

    axes.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes.set_ylim((0, 1.05))
    axes.set_ylabel("True Positive Rate")
    axes.set_xlim((0, 1.0))
    axes.set_xlabel("False Positive Rate")
    axes.set_title("One vs all")
    axes.legend()
    return axes


def auc_per_experiment(somiter_data: dict) -> dict:
    """Get average AUC values over groups for each experiment."""
    aucs = {
        k: np.mean(list(_auc_one_vs_all(v).values()))
        for k, v in somiter_data.items()
    }
    return aucs


def auc_per_group(somiter_data: dict) -> dict:
    """Get average AUC values over experiments for each group."""
    auc_stats = {
        k: v / len(somiter_data) for k, v in reduce(
            lambda x, y: {k: v + y[k] for k, v in x.items()},
            [_auc_one_vs_all(d) for d in somiter_data.values()]
        ).items()
    }
    return auc_stats


def auc_dfs(somiter_data: dict) -> dict:
    """Get auc dataframes with dict key as outermost key."""
    auc_dfs = {
        k: _auc(v) for k, v in somiter_data.items()
    }
    auc_df = pd.concat(auc_dfs.values(), keys=auc_dfs.keys())
    return auc_df


def _auc(data: pd.DataFrame) -> pd.DataFrame:
    """ROC and AUC calculations. This is done one-vs-all for each call.
    macro - take for each group and average after
    micro - count each sample
    """
    scores = df_prediction_cols(data)
    binarizer = LabelBinarizer()
    true_labels = binarizer.fit(scores.columns).transform(data["group"].values)

    auc_vals = {
        "auc": {
            m: roc_auc_score(true_labels, scores, average=m)
            for m in ["macro", "micro"]
        }
    }
    return pd.DataFrame.from_dict(auc_vals, orient="index")


def _auc_one(data: pd.DataFrame, pos_label) -> float:
    """Calculate ROC AUC in a one-vs-all manner."""
    return roc_auc_score(data["group"] == pos_label, data[pos_label])


def _auc_one_vs_all(data: pd.DataFrame) -> dict:
    """Calculate one-vs-all auc for all groups in dataframe."""
    return {
        group: _auc_one(data, group)
        for group in data["group"].unique()
    }


def _roc(data: pd.DataFrame) -> dict:
    """Calculate ROC curve. Initially only using macro approach.
    """
    results = {}
    for name in data["group"].unique():
        fpr, tpr, thres = roc_curve(
            data["group"], data[name], pos_label=name
        )
        results[name] = pd.DataFrame.from_dict(
            {"fpr": fpr, "tpr": tpr, "thres": thres}
        )
    result = pd.concat(results.values(), keys=results.keys())
    result.index.rename("positive", level=0, inplace=True)
    return result


def _interpolate_roc_curves(roc_curves: pd.DataFrame):
    """Interpolate a number of roc curves in a dataframe."""
    # set an index for binning the values
    bin_index = pd.interval_range(start=0, end=1, periods=100)

    roc_curves["bin"] = pd.cut(roc_curves["fpr"], bin_index)
    roc_curves.set_index("bin", append=True, inplace=True)
    roc_mean = roc_curves.mean(
        level=["positive", "bin"]
    ).interpolate()
    roc_mean["std"] = roc_curves["tpr"].std(
        level=["positive", "bin"]
    ).interpolate()
    return roc_mean


def _prediction_dict_to_roc_df(pdict: dict) -> pd.DataFrame:
    """Get dataframe with AUC values from dict with prediction dataframes.
    Interpolate data for individual experiments to get avg for a single
    group.
    """
    roc_curves = pd.concat(
        [_roc(d) for d in pdict.values()], keys=pdict.keys()
    )
    roc_curves.index.rename("som", level=0, inplace=True)
    roc_mean = _interpolate_roc_curves(roc_curves)
    return roc_mean
