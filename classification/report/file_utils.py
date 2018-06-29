"""Utilities to get folders and read files"""
import os
import re
import glob
import json
from math import sqrt
from datetime import datetime
from functools import reduce

import pandas as pd


PREDICTION_SUFFIX = "predictions.csv"


def row_load_stats(row: pd.Series) -> pd.Series:
    """Load some information from avg_stats.csv into the dataframe."""
    statpath = os.path.join(row["path"], "overview_plots/avg_stats.csv")
    if not os.path.exists(statpath):
        return None
    loaded = pd.read_table(statpath, index_col=0, sep=",")

    return loaded.loc["mean", "f1"], loaded.loc["std", "f1"] ** 2


def add_avg_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Add average stats to a dataframe based on loading avg_stats from path
    column."""
    data["f1"] = data.apply(row_load_stats, axis=1)
    data = data.dropna().reset_index(drop=True)
    return data


def load_experiments(path: str) -> pd.DataFrame:
    """Parse dirnames into a table structure.
    Directory names need to have the following structure:
        <SET_NAME><EXP_NAME><o:selected><YYYYMMDD_HHMM>
    Args:
        path: Base path for searching directories containing projects
    Returns:
        Information parsed from the directory names.
    """
    with os.scandir(path) as pdir:
        filenames = [p.name for p in pdir if p.is_dir()]

    fdict = [
        {
            **k,
            "name": k["name"].replace(k["set"]+"_", ""),
        }
        for k in
        [
            {
                "set": m[1],
                "name": k.replace("_selected", "").replace("_"+m[2], ""),
                "time": datetime.strptime(m[2], "%Y%m%d_%H%M"),
                "type": "Selected" if "selected" in k else "Random",
                "path": os.path.join(path, k)
            }
            for k, m in
            [
                (
                    k,
                    re.match(r"([^\W_]+_[^\W_]+)_[^\W]+_(\d+_\d+)", k),
                )
                for k in filenames
            ]
            if m
        ]
    ]

    return pd.DataFrame.from_dict(fdict, orient="columns")


def get_prediction_paths(path: str) -> [str]:
    return glob.glob("{}/*/*{}".format(path, PREDICTION_SUFFIX))


def load_predictions(paths: [str]) -> dict:
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
        for p in paths
    }
    return loaded


def add_prediction_info(experiments: pd.DataFrame) -> pd.DataFrame:
    """Add prediction information to dataframe and drop rows without this
    info."""
    # only work on most recent version of experiment
    most_recent = experiments.groupby(["set", "name", "type"]).apply(
        lambda d: d.loc[d["time"].idxmax()]
    )
    most_recent["predictions"] = most_recent["path"].apply(
        lambda p: get_prediction_paths(p)
    )
    most_recent = most_recent.loc[
        most_recent["predictions"].str.len().astype("bool")
    ]

    return most_recent


def load_json(path: str):
    """Load a json file."""
    with open(path) as jsfile:
        data = json.load(jsfile)
    return data

def avg_meta(first, second):
    """Combine information from different runs for the same experiment."""
    averaged = ["note", "command", "group_names"]
    # enforce sameness
    for key in averaged:
        assert str(first[key]) == str(second[key])

    return {
        k: first[k]
        for k in averaged
    }

def load_metadata(path: str) -> dict:
    """Load metadata into a dict structure."""
    paths = glob.glob("{}/*/*{}".format(path, "info.json"))
    metadatas = [
        load_json(p) for p in paths
    ]
    metadata = reduce(avg_meta, metadatas)
    return metadata
