"""Utilities to get folders and read files"""
import os
from pathlib import Path
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
    exp_sets = [
        (expset, exp) for expset in Path(path).iterdir() if expset.is_dir()
        for exp in expset.iterdir() if exp.is_dir()
    ]
    exp_sets += [
        (expset, expset) for expset in Path(path).iterdir()
        if expset.is_dir()
        and any([f.name.endswith(".csv") for f in expset.iterdir()])
    ]
    exp_dicts = []
    for expset, exp in exp_sets:
        match = re.match(r"(\w+?_)(\d+_)?(selected_)?(\d+_\d+)", exp.name)
        if not match:
            continue
        name = match[1] or expset.name
        iteration = int(match[2].strip("_")) if match[2] else 0
        edict = {
            "set": expset.name,
            "name": name.strip("_"),
            "iteration": iteration,
            "time": datetime.strptime(match[4], "%Y%m%d_%H%M"),
            "type": "selected" if match[3] else "random",
            "path": str(exp),
        }
        exp_dicts.append(edict)

    exp_df = pd.DataFrame.from_dict(exp_dicts, orient="columns")
    return exp_df


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


def filter_infiltration(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Filter infiltration rate for other assessments."""



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


def load_metadata(path: str) -> [dict]:
    """Load metadata into a dict structure."""
    paths = glob.glob("{}/*/*{}".format(path, "info.json"))
    return [
        load_json(p) for p in paths
    ]


def load_avg_metadata(path: str) -> dict:
    """Return average reduced metadata."""
    return reduce(avg_meta, load_metadata(path))
