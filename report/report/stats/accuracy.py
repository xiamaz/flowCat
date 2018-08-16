"""Accuracy metrics for classification."""
import numpy as np
import pandas as pd

from .. import file_utils


def acc_table_dfs(pdata: dict):
    return _pdict_transform(pdata, acc_table)


def acc_table(data: pd.DataFrame):
    """Create accuracy table with micro and macro accuriacies."""
    accs = {
        "t1": top1_uncertainty,
        "t2": top2,
    }
    data = {
        k: {m: avg_accuracy(data, v, avg_type=m) for m in ["macro", "micro"]}
        for k, v in accs.items()
    }
    acc_df = pd.DataFrame.from_dict(data, orient="index")
    return acc_df

def avg_accuracy(data: pd.DataFrame, fun, avg_type="macro", *args, **kwargs):
    """Calculate average accuracy between classes either as an average
    of classes or an individual average by multiplying the accuracies with
    number of cases."""
    group_accs = fun(data, *args, **kwargs)["correct"]

    if avg_type == "macro":
        mean = np.mean(group_accs)
    elif avg_type == "micro":
        group_sizes = data.groupby("group").size()
        mean = sum(group_accs * group_sizes) / sum(group_sizes)
    else:
        raise TypeError("Unknown average type.")

    return mean


def top2_dfs(pdata):
    """Top2 accuracy table for prediction dicts."""
    return _pdict_transform(pdata, top2)


def top_combined_dfs(pdata):
    """Combination of top accuracies table for prediction dicts."""
    return _pdict_transform(pdata, top_combined)


def top1_uncertainty_dfs(pdata, *args, **kwargs):
    """Top1 with uncertainty for prediciton dicts."""
    return _pdict_transform(pdata, top1_uncertainty, *args, **kwargs)


def df_get_predictions_t1(data: pd.DataFrame) -> pd.DataFrame:
    """Get dataframe of prediction names with certainties for each case."""

    pdata = file_utils.df_prediction_cols(data)
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


def top_combined(data: pd.DataFrame) -> pd.DataFrame:
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


def top2(data: pd.DataFrame, normal_t1=True) -> pd.Series:
    """Get the top 2 classifications, while leaving normal as top 1."""
    results = {}

    pdata = file_utils.df_prediction_cols(data)
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


def top1_uncertainty(data: pd.DataFrame, threshold=0.0) -> pd.DataFrame:
    """Adding a threshold below which cases are sorted to uncertain class."""
    results = {}

    pdata = file_utils.df_prediction_cols(data)

    preds = np.argmax(pdata.values, axis=1)
    maxvals = pdata.values[np.arange(len(preds)), preds]
    gnums = np.vectorize(
        lambda x: list(pdata.columns).index(x)
    )(data["group"])
    rdata = np.stack((gnums, preds, maxvals), axis=1)

    # all_acc_dict = {"corr": 0, "all": 0}
    for i, name in enumerate(pdata.columns):
        group_data = rdata[rdata[:, 0] == i]
        correct = group_data[:, 1] == i
        certain = group_data[:, 2] >= threshold

        # all_acc_dict["corr"] += sum(correct)
        # all_acc_dict["all"] += group_data.shape[0]

        results[name] = [
            sum(correct & certain) / group_data.shape[0],
            sum(~certain) / group_data.shape[0],
            sum((~correct) & certain) / group_data.shape[0],
        ]
    # print("Overall acc: {}".format(all_acc_dict["corr"] / all_acc_dict["all"]))
    return pd.DataFrame.from_dict(
        results, orient="index", columns=["correct", "uncertain", "incorrect"]
    )


def _pdict_transform(pdata, fun, *args, **kwargs):
    """Transform dict of prediction dataframes into a multiindex dataframe
    containing results."""
    tdata = {k: fun(v, *args, **kwargs) for k, v in pdata.items()}
    trans_df = pd.concat(tdata.values(), keys=tdata.keys())
    return trans_df
