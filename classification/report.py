"""Create simple overviews over created experiments."""
import os
import re
import sys
from math import sqrt
from functools import reduce
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

import altair as alt


def get_files(path: str, threshold: int = 1) -> dict:
    """Parse dirnames into a dictionary structure."""
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


def load_stats(row: pd.Series) -> pd.Series:
    statpath = os.path.join(row["path"], "overview_plots/avg_stats.csv")
    if not os.path.exists(statpath):
        return None
    loaded = pd.read_table(statpath, index_col=0, sep=",")

    return loaded.loc["mean", "f1"], loaded.loc["std", "f1"] ** 2


def avg_stat(data: pd.DataFrame):

    count = data.shape[0]

    f1, var = tuple(map(
        lambda x: x/count,
        reduce(
            lambda ab, xy: (ab[0]+xy[0], ab[1]+xy[1]), data["f1"], (0, 0)
        )
    ))
    std = sqrt(var)

    return pd.Series(data=[
        count, f1, std
    ], index=[
        "count", "f1", "std"
    ])


def group_stats(data: pd.DataFrame):
    data["f1"] = data.apply(load_stats, axis=1)
    data = data.dropna().reset_index(drop=True)
    resp = data.groupby(["set", "name", "type"]).apply(avg_stat)
    return resp


def texclean(raw: str) -> str:
    """Escape special signs in latex."""
    return raw.replace("_", r"\_")


def df_tabulate(data: pd.DataFrame, level: int = 0) -> dict:
    """Create deduplicated dataframe structure."""
    result = {}
    max_level = len(data.index.names)
    for name, gdata in data.groupby(level=level):
        if level < max_level-1:
            result[name] = df_tabulate(gdata, level+1)
        else:
            result[name] = [
                " & ".join([texclean(str(t)) for t in r]) + r" \\"
                for r in gdata.apply(
                    lambda r: list(r.values), axis=1
                )
            ]

    output = [
        r for k, v in result.items()
        for r in [texclean(k) + " & " + v[0]] + ["  & " + r for r in v[1:]]
    ]
    return output

def df_to_table(data: pd.DataFrame, spec: str = ""):
    """Create latex table fragment from pandas data with smart printing
    of multiindexes."""
    # create title
    index_depth = len(data.index.names)
    if not spec:
        spec = "l"*(index_depth+data.shape[1])
    outtext = [
        r"\begin{{tabular}}{{{}}}".format(spec),
        r"\toprule",
    ]

    if index_depth > 1:
        names = [
            texclean(n) for n in list(data.index.names) + list(data.columns)
        ]
        outtext += [
            r"\multicolumn{{{}}}{{c}}{{Index}} \\".format(index_depth),
            r"\cmidrule(r){{1-{}}}".format(index_depth),
            " & ".join(names) + r" \\",
            r"\midrule",
        ]

    # create data
    outtext += list(
        df_tabulate(data)
    )

    # create footer
    outtext += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(outtext)


def plot_clustering_experiments(
        path: str = "report",
        filename: str = "experiment_overview.png"
):
    """Plot experiment information using altair."""
    data = get_files("output/clustering")
    counts = pd.DataFrame(
        data.groupby(["set", "name", "type"]).size(), columns=["count"]
    ).reset_index()
    counts = counts.loc[counts["count"] > 1]
    charts = []

    for sname, sdata in counts.groupby("set"):
        parts = []
        for ename, edata in sdata.groupby("name"):
            base = alt.Chart(edata).mark_bar().encode(
                y=alt.Y("type:N", axis=alt.Axis(title=ename)),
                x=alt.X(
                    "count:Q",
                    axis=alt.Axis(title=""),
                    scale=alt.Scale(domain=(0, 10))
                ),
                color="type:N",
            )
            parts.append(base)
        part = alt.vconcat(*parts)
        part.title = sname
        charts.append(part)

    chart = alt.hconcat(*charts).configure(
        axis=alt.AxisConfig(
            titleAngle=0,
            titleLimit=0,
            titleAlign="left",
            titleX=0,
            titleY=0,
        ),
        title=alt.VgTitleConfig(
            offset=20
        )
    )
    cpath = os.path.join(path, filename)
    chart.save(cpath)


def table_stats(
        path: str = "report",
        filename: str = "result.tex"
) -> None:
    files = get_files("output/classification", threshold=0)
    data = group_stats(files).round(2)

    # filtering data
    data = data.loc[
        ~data.index.get_level_values(
            "name"
        ).str.contains("all_groups|more_merged")
    ]

    data["count"] = data["count"].astype("int32").apply(str)
    table_string = df_to_table(data)
    tpath = os.path.join(path, filename)
    with open(tpath, "w") as tfile:
        tfile.write(table_string)


OUTPATH = sys.argv[1]
os.makedirs(OUTPATH, exist_ok=True)

table_stats(OUTPATH)
plot_clustering_experiments(OUTPATH)
