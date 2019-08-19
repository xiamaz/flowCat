"""Common utilies for working with intervals.
"""
from __future__ import annotations
from typing import Union
from functools import reduce
import pandas as pd


class IntervalException(Exception):
    pass


def create_interval(interval: Union[tuple, pd.Interval]):
    if isinstance(interval, pd.Interval):
        return interval
    lower, upper = interval
    if lower is None:
        lower = -float("inf")
    if upper is None:
        upper = float("inf")
    return pd.Interval(lower, upper, closed="neither")


def outer_interval(first: pd.Interval, second: pd.Interval) -> pd.Interval:
    """Merge two intervals, so that their range will reflect the entire range."""
    if not first.overlaps(second):
        raise IntervalException("Intervals do not overlap")

    left_a = first.left
    left_b = second.left
    if left_a < left_b:
        left = left_a
        left_close = first.closed_left
    elif left_a > left_b:
        left = left_b
        left_close = second.closed_left
    else:
        left = left_a
        left_close = first.closed_left or second.closed_left

    right_a = first.right
    right_b = second.right
    if right_a > right_b:
        right = right_a
        right_close = first.closed_right
    elif right_a < right_b:
        right = right_b
        right_close = second.closed_right
    else:
        right = right_a
        right_close = first.closed_right or second.closed_right

    if left_close and right_close:
        closed = "both"
    elif left_close:
        closed = "left"
    elif right_close:
        closed = "right"
    else:
        closed = "neither"

    new_interval = pd.Interval(left, right, closed)
    return new_interval


def outer_intervals(intervals: Iterable[pd.Interval]):
    return reduce(lambda x, y: outer_interval(x, y), intervals)


def inner_interval(first: pd.Interval, second: pd.Interval) -> pd.Interval:
    """Merge two interval, so that the resulting interval will be smaller or equal than both."""
    if not first.overlaps(second):
        raise IntervalException("Intervals do not overlap")

    left_a = first.left
    left_b = second.left
    if left_a > left_b:
        left = left_a
        left_close = first.closed_left
    elif left_a < left_b:
        left = left_b
        left_close = second.closed_left
    else:
        left = left_a
        left_close = first.closed_left and second.closed_left

    right_a = first.right
    right_b = second.right
    if right_a < right_b:
        right = right_a
        right_close = first.closed_right
    elif right_a > right_b:
        right = right_b
        right_close = second.closed_right
    else:
        right = right_a
        right_close = first.closed_right and second.closed_right

    if left_close and right_close:
        closed = "both"
    elif left_close:
        closed = "left"
    elif right_close:
        closed = "right"
    else:
        closed = "neither"

    new_interval = pd.Interval(left, right, closed)
    return new_interval


def inner_intervals(intervals: Iterable[pd.Interval]):
    return reduce(lambda x, y: inner_interval(x, y), intervals)


def series_in_interval(series: pd.Series, interval: pd.Interval):
    if interval.closed == "both":
        contained = (series >= interval.left) & (series <= interval.right)
    elif interval.closed == "right":
        contained = (series > interval.left) & (series <= interval.right)
    elif interval.closed == "left":
        contained = (series >= interval.left) & (series < interval.right)
    else:
        contained = (series > interval.left) & (series < interval.right)
    return contained
