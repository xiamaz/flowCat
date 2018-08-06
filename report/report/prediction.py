import typing

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve

from .base import Reporter
from .file_utils import load_metadata


def df_prediction_cols(data: pd.DataFrame) -> np.matrix:
    predictions = data.select_dtypes("number").drop(
        "infiltration", axis=1
    ).astype("float32")
    return predictions


def prediction_ranks(row: pd.Series) -> [str]:
    """Get a list of predictions in descending order."""
    labels = row.index[np.argsort(row)[::-1]]
    return labels


def predict(row: pd.Series, threshold=0):
    """Get prediction or return uncertain if below threshold."""
    pred = row.idxmax()
    return pred if row[pred] >= threshold else "uncertain"


def top2(data: pd.DataFrame, normal_t1=True) -> pd.Series:
    """Get the top 2 classifications, while leaving normal as top 1."""
    results = {}

    pdata = df_prediction_cols(data)
    group_names = list(pdata.columns)
    groups = zip(list(range(len(pdata.columns))), group_names)

    gnums = np.vectorize(lambda x: group_names.index(x))(data["group"])
    preds = np.argsort(pdata.values, axis=1)

    t1 = preds[:, -1] == gnums
    t2 = preds[:, -2] == gnums

    nnormal = list(pdata.columns).index("normal")
    normal = preds[:, -1] == nnormal

    if normal_t1:
        groups = [(i, n) for i, n in groups if n != "normal"]
        corr = sum(normal[gnums == nnormal]) / sum(gnums == nnormal)
        results["normal"] = [corr, 0, 1 - corr]

    for i, name in groups:
        selection = gnums == i
        group_size = sum(selection)
        if normal_t1:
            selection = selection & (~normal)

        corr = sum(t1[selection] | t2[selection]) / group_size
        results[name] = [corr, 0, 1 - corr]

    return pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["correct", "uncertain", "incorrect"]
    ).sort_index()


def df_get_predictions_t1(data: pd.DataFrame) -> pd.DataFrame:
    """Get dataframe of prediction names with certainties for each case."""

    pdata = df_prediction_cols(data)
    preds = np.argmax(pdata.values, axis=1)
    maxvals = pdata.values[np.arange(len(preds)), preds]
    pred_names = np.vectorize(lambda x: pdata.columns[x])(preds)
    rdata = pd.DataFrame(
        {
            "group": data["group"].values,
            "prediction": pred_names,
            "certainty": maxvals,
        },
        index=data.index
    )
    return rdata


def top1_uncertainty(data: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    """Adding a threshold below which cases are sorted to uncertain class."""
    results = {}

    pdata = df_prediction_cols(data)

    preds = np.argmax(pdata.values, axis=1)
    maxvals = pdata.values[np.arange(len(preds)), preds]
    gnums = np.vectorize(
        lambda x: list(pdata.columns).index(x)
    )(data["group"])
    rdata = np.stack((gnums, preds, maxvals), axis=1)

    all_acc_dict = {"corr": 0, "all": 0}
    for i, name in enumerate(pdata.columns):
        group_data = rdata[rdata[:, 0] == i]
        correct = group_data[:, 1] == i
        certain = group_data[:, 2] >= threshold

        all_acc_dict["corr"] += sum(correct)
        all_acc_dict["all"] += group_data.shape[0]

        results[name] = [
            sum(correct & certain) / group_data.shape[0],
            sum(~certain) / group_data.shape[0],
            sum((~correct) & certain) / group_data.shape[0],
        ]
    print("Overall acc: {}".format(all_acc_dict["corr"] / all_acc_dict["all"]))
    return pd.DataFrame.from_dict(
        results, orient="index", columns=["correct", "uncertain", "incorrect"]
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


def df_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Create a dataframe containing different metrics in rows and df as
    row."""
    testdict = {
        "top1_t04": lambda d: top1_uncertainty(d, threshold=0.4),
        "top1_t06": lambda d: top1_uncertainty(d, threshold=0.6),
        "top1_t08": lambda d: top1_uncertainty(d, threshold=0.8),
        "top2_n+": lambda d: top2(d, normal_t1=False),
        "top2_n-": lambda d: top2(d, normal_t1=True),
    }
    result = pd.concat(
        [fun(data) for fun in testdict.values()], keys=testdict.keys()
    ).swaplevel(0, 1).sort_index()
    return result


class Prediction(Reporter):
    """Create prediction analysis results from classification."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_experiment(self, row, path):
        """Create plots for the given experiment."""

        pname = "{}_{}_{}".format(*row.name)

        plotpath = os.path.join(path, "predictions_{}".format(pname))
        somiter_data = load_predictions(row["predictions"])
        metadata = load_metadata(row["path"])[0]

        rocpath = plotpath + "_auc.png"
        if not os.path.exists(rocpath):
            fig = plot_avg_roc_curves(somiter_data)
            fig.savefig(rocpath, dpi=200)
        else:
            print("{} already exists. Not recreating".format(rocpath))

        chartpath = plotpath + "_stats.png"
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

        prediction_data = add_prediction_info(self.classification_files)
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
