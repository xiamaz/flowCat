"""
Transform operations for overview generation.
"""
from functools import reduce
from math import sqrt
import pandas as pd


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
