"""Currently only supports S3 objects."""
import abc
import os
import pathlib
import json
import collections
import pickle
from urllib.parse import urlparse
import logging
import contextlib
import time
import datetime
import fnmatch
from functools import wraps

import toml
import numpy as np
import pandas as pd
import boto3
import joblib

from . import mappings


LOGGER = logging.getLogger(__name__)
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class FCEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=E0202
        if type(obj) in mappings.PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        if isinstance(obj, URLPath):
            return {"__urlpath__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def as_fc(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(mappings.PUBLIC_ENUMS[name], member)
    elif "__urlpath__" in d:
        return URLPath(d["__urlpath__"])
    else:
        return d


def str_to_date(strdate):
    return datetime.datetime.strptime(strdate, "%Y-%m-%d").date()


class URLPath(pathlib.PosixPath):
    __slots__ = (
        "_scheme",
        "_netloc"
    )

    @classmethod
    def _from_parts(cls, args, *other, **kwargs):
        first = args[0]
        if isinstance(first, URLPath):
            scheme = first._scheme
            netloc = first._netloc
            path = first
        else:
            urlpath = urlparse(str(first))
            scheme = urlpath.scheme
            netloc = urlpath.netloc
            path = urlpath.path

        self = super()._from_parts((path, *args[1:]), *other, **kwargs)
        self._scheme = scheme
        self._netloc = netloc
        return self

    def _from_parsed_parts(self, *args, **kwargs):
        obj = super()._from_parsed_parts(*args, **kwargs)
        obj._scheme = self._scheme
        obj._netloc = self._netloc
        return obj

    def mkdir(self, mode=0o777, exist_ok=True, parents=True):
        return super().mkdir(mode=mode, exist_ok=exist_ok, parents=parents)

    def open(self, *args, **kwargs):
        self.parent.mkdir(exist_ok=True, parents=True)
        return super().open(*args, **kwargs)

    def __str__(self):
        if self._scheme:
            return f"{self._scheme}://{self._netloc}{super().__str__()}"
        else:
            return super().__str__()

    def __add__(self, other):
        return self.__class__(str(self) + str(other))

    def __radd__(self, other):
        return self.__class__(str(other) + str(self))


def load_json(path: URLPath):
    """Load json data from a path as a simple function."""
    with path.open("r") as jspath:
        data = json.load(jspath, object_hook=as_fc)
    return data


def save_json(data, path: URLPath):
    """Write json data to a file as a simple function."""
    with path.open("w") as jsfile:
        json.dump(data, jsfile, cls=FCEncoder)


def load_pickle(path: URLPath):
    with path.open("rb") as pfile:
        data = pickle.load(pfile)
    return data


def save_pickle(data, path: URLPath):
    """Write data to the given path as a pickle."""
    with path.open("wb") as pfile:
        pickle.dump(data, pfile)


def load_joblib(path: URLPath):
    return joblib.load(str(path))


def save_joblib(data, path: URLPath):
    path.parent.mkdir()
    joblib.dump(data, str(path))


def to_json(data):
    return json.dumps(data, indent=4)


def load_csv(path, index_col=0):
    data = pd.read_csv(str(path), index_col=index_col)
    return data


def save_csv(data: pd.DataFrame, path: URLPath):
    path.parent.mkdir(exist_ok=True, parents=True)
    data.to_csv(path)


def create_stamp():
    """Create timestamp usable for filepaths"""
    stamp = datetime.datetime.now()
    return stamp.strftime(TIMESTAMP_FORMAT)


@contextlib.contextmanager
def timer(title):
    """Take the time for the enclosed block."""
    time_a = time.time()
    yield
    time_b = time.time()

    time_diff = time_b - time_a
    print(f"{title}: {time_diff:.3}s")


def time_generator_logger(generator, rolling_len=20):
    """Time the given generator.
    Args:
        generator: Will be executed and results are passed through.
        rolling_len: Number of last values for generation of average time.

    Returns:
        Any value returned from the given generator.
    """
    circ_buffer = collections.deque(maxlen=rolling_len)
    time_a = time.time()
    for res in generator:
        time_b = time.time()
        time_d = time_b - time_a
        circ_buffer.append(time_d)
        time_rolling = np.mean(circ_buffer)
        print(f"Training time: {time_d}s Rolling avg: {time_rolling}s")
        time_a = time_b
        yield res


def df_get_count(data, tubes):
    """Get count information from the given dataframe with labels as index.
    Args:
        data: dict of dataframes. Index will be used in the returned count dataframe.
        tubes: List of tubes as integers.
    Returns:
        Dataframe with labels as index and ratio of availability in the given tubes as value.
    """
    counts = None
    for tube in tubes:
        count = pd.DataFrame(
            1, index=data[tube].index, columns=["count"])
        count.reset_index(inplace=True)
        count.set_index("label", inplace=True)
        count = count.loc[~count.index.duplicated(keep='first')]
        count.drop("group", axis=1, inplace=True, errors="ignore")
        if counts is None:
            counts = count
        else:
            counts = counts.add(count, fill_value=0)
    counts = counts / len(tubes)
    return counts


def add_logger(log, handlers, level=logging.DEBUG):
    """Add logger or name of a logger to a list of handlers."""
    if isinstance(log, str):
        log = logging.getLogger(log)
    elif not isinstance(log, logging.Logger):
        raise TypeError("Wrong type for log")

    log.setLevel(level)
    for handler in handlers:
        log.addHandler(handler)


def create_handler(handler, fmt=LOGGING_FORMAT, level=logging.DEBUG):
    """Add a logging formatter to the given Handler."""
    handler.setLevel(level)
    if not isinstance(fmt, logging.Formatter):
        fmt = logging.Formatter(fmt)
    handler.setFormatter(fmt)
    return handler


def load_labels(path):
    """Load list of labels. Either in .json, .p (pickle) or .txt format.
    Args:
        path: path to label file.
    """
    if not path:
        return path

    path = URLPath(path)
    try:
        labels = load_file(path)
    except TypeError:
        # try loading as simple txt file instead
        with open(str(path), "r") as f:
            labels = [l.strip() for l in f]
    return labels
