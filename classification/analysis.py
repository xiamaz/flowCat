# flake8: noqa
import os
import re
import glob

from functools import reduce

import numpy as np
import pandas as pd

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import cm
from matplotlib.figure import Figure

from collections import Counter

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve


SUFFIX = "predictions.csv"


def get_folders(path, pattern: str) -> [str]:
    """Return all dirs in directory with names matching the given pattern.

    >>> get_folders("output/classification", "normal")
    """
    matched = glob.glob("{}/*{}*/".format(path, pattern))
    return matched


def load_predictions(path: str) -> [pd.DataFrame]:
    """Return a list with all predictions for the given classification
    loaded.

    Directly load a directory with csv in subdirs.
    >>> load_predictions("output/classification/mini_20180606_0733/")

    Load by getting folders with get_folders first.
    >>> t = get_folders("output/classification", "selected_normal")[0]
    >>> load_predictions(t)
    """
    loaded = {
        p.split("/")[-2]: pd.read_table(p, delimiter=",", index_col=0)
        for p in glob.glob("{}/*/*{}".format(path, SUFFIX))
    }
    return loaded


def df_prediction_cols(data: pd.DataFrame) -> np.matrix:
    predictions = data.select_dtypes("number").drop(
       "infiltration", axis=1).astype("float32")
    return predictions


def prediction_ranks(row: pd.Series) -> [str]:
    """Get a list of predictions in descending order."""
    labels = row.index[np.argsort(row)[::-1]]
    return labels


def predict(row: pd.Series, threshold=0):
    """Get prediction or return uncertain if below threshold."""
    pred = row.idxmax()
    return pred if row[pred] >= threshold else "uncertain"


def top2(data: pd.DataFrame) -> pd.Series:
    """Get the top 2 classifications for each group."""
    results = {}
    for name, group in data.groupby("group"):
        gpreds = df_prediction_cols(group)
        preds = group.apply(lambda r: name in prediction_ranks(r)[:2], axis=1)
        acc = sum(preds) / group.shape[0]
        results[name] = [acc]
    return pd.DataFrame.from_dict(results, orient="index", columns=["correct"])


def top2_sans_normal(data: pd.DataFrame) -> pd.Series:
    """Get the top 2 classifications, while leaving normal as top 1."""
    results = {}
    for name, group in data.groupby("group"):
        gpreds = df_prediction_cols(group)
        if name != "normal":
            preds = gpreds.apply(
                lambda r: name in prediction_ranks(r)[:2], axis=1
            )
            n_preds = gpreds.apply(
                lambda r: "normal" == prediction_ranks(r)[0], axis=1
            )
            preds = preds & (~n_preds)
        else:
            preds = gpreds.apply(
                lambda r: "normal" == prediction_ranks(r)[0], axis=1
            )

        acc = sum(preds) / group.shape[0]
        results[name] = [acc]
    return pd.DataFrame.from_dict(results, orient="index", columns=["correct"])


def top1_uncertainty(data: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    """Adding a threshold below which cases are sorted to uncertain class."""
    results = {}
    for name, group in data.groupby("group"):
        gpreds = df_prediction_cols(group)
        pred = gpreds.apply(lambda r: predict(r, threshold), axis=1)
        correct = sum(pred == name)
        uncert = sum(pred == "uncertain")

        results[name] = [
            correct/(group.shape[0]-uncert), uncert/group.shape[0]
        ]
    return pd.DataFrame.from_dict(
        results, orient="index", columns=["correct", "uncertain"]
    )


def auc(data: pd.DataFrame) -> pd.DataFrame:
    """ROC and AUC calculations. This is done one-vs-all for each call.
    macro - take for each group and average after
    micro - count each sample
    """
    scores = df_prediction_cols(data)
    binarizer = LabelBinarizer()
    true_labels = binarizer.fit(scores.columns).transform(data["group"].values)

    auc = {
        "macro": [roc_auc_score(true_labels, scores, average="macro")],
        "micro": [roc_auc_score(true_labels, scores, average="micro")]
    }
    return pd.DataFrame.from_dict(auc, columns=["auc"], orient="index")


def auc_one(data: pd.DataFrame, pos_label) -> float:
    """Calculate ROC AUC in a one-vs-all manner."""
    return roc_auc_score(data["group"] == pos_label, data[pos_label])


def auc_one_vs_all(data: pd.DataFrame) -> dict:
    """Calculate one-vs-all auc for all groups in dataframe."""
    return {
        group: auc_one(data, group)
        for group in data["group"].unique()
    }


def roc(data: pd.DataFrame) -> dict:
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


def plot_roc(roc_data: dict, auc: dict, ax: "Axes") -> Figure:
    colors = cm.tab20.colors #pylint: disable=no-member
    for i, (name, data) in enumerate(roc_data.groupby(level=0)):
        ax.plot(
            data["fpr"],
            data["tpr"],
            color=colors[i*2], lw=1,
            label="{} (AUC {:.2f})".format(name, auc[name])
        )
        ax.fill_between(
            data["fpr"],
            data["tpr"]-data["std"],
            data["tpr"]+data["std"],
            color=colors[i*2+1],
            alpha = 1-(1/len(auc)*i),
        )

    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_ylim((0, 1.05))
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim((0, 1.0))
    ax.set_xlabel("False Positive Rate")
    ax.set_title("One vs all")
    ax.legend()
    return ax


def df_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Create a dataframe containing different metrics in rows and df as
    row."""
    testdict = {
        "top1_t04": lambda d: top1_uncertainty(d, threshold=0.4),
        "top1_t06": lambda d: top1_uncertainty(d, threshold=0.6),
        "top1_t08": lambda d: top1_uncertainty(d, threshold=0.8),
    }
    result = pd.concat(
        [fun(data) for fun in testdict.values()], keys=testdict.keys()
    ).swaplevel(0, 1).sort_index()
    return result


def avg_stats(somiter_data: dict, ax) -> pd.DataFrame:
    """Average df results from multiple dataframes."""
    som_df = pd.concat(
        [df_stats(d) for d in somiter_data.values()], keys=somiter_data.keys()
    )
    mean = som_df.mean(level=[1, 2])
    std = som_df.std(level=[1, 2])
    ax = mean.plot.bar(yerr=std, ax=ax)
    return ax


def avg_auc(somiter_data: dict, ax: "Axes"):
    """Create a figure containing average auc data."""
    roc_curves = pd.concat(
        [roc(d) for d in somiter_data.values()], keys=somiter_data.keys()
    )
    roc_curves.index.rename("som", level=0, inplace=True)

    bin_index = pd.interval_range(0, 1, 100)
    roc_curves["bin"] = pd.cut(roc_curves["fpr"], bin_index)
    roc_curves.set_index("bin", append=True, inplace=True)
    roc_mean = roc_curves.mean(
        level=["positive", "bin"]
    ).interpolate()
    roc_mean["std"] = roc_curves["tpr"].std(
        level=["positive", "bin"]
    ).interpolate()
    auc_stats = {
        k: v/len(somiter_data) for k, v in reduce(
            lambda x, y: {k: v+y[k] for k, v in x.items()},
            [auc_one_vs_all(d) for d in somiter_data.values()]
        ).items()
    }
    ax = plot_roc(roc_mean, auc_stats, ax=ax)
    return ax


def plot_experiment(pattern: str, somiter_data: dict):
    """Return statistics average over multiple iterations."""
    fig = Figure()

    # average AUC plots
    ax = fig.add_subplot(211)
    avg_auc(somiter_data, ax)

    ax = fig.add_subplot(212)
    avg_stats(somiter_data, ax)

    fig.set_size_inches(8, 16)
    fig.suptitle(pattern)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FigureCanvas(fig)
    fig.savefig("analysis/{}".format(pattern), dpi=200)


def main():
    experiments = {
        ex: get_folders("output/classification", ex)
        for ex in
        [
            "initial_comp_selected_normal_dedup",
            "initial_comp_selected_indiv_pregating_dedup"
        ]
    }

    # create dict of experiment name and list of loaded dataframes
    predictions = {
        ex: {
            k: v for e in folders for k, v in load_predictions(e).items()
        }
        for ex, folders in experiments.items()
    }

    for experiment, dataframes in predictions.items():
        plot_experiment(experiment, dataframes)
