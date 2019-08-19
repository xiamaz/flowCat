import json

import pickle
import joblib
import pandas as pd

from flowcat import mappings
from .urlpath import URLPath


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
    with path.open("wb") as handle:
        joblib.dump(data, handle)


def to_json(data):
    return json.dumps(data, indent=4)


def load_csv(path, index_col=0):
    data = pd.read_csv(str(path), index_col=index_col)
    return data


def save_csv(data: pd.DataFrame, path: URLPath):
    path.parent.mkdir(exist_ok=True, parents=True)
    data.to_csv(path)
