"""Currently only supports S3 objects."""
import abc
import os
import pathlib
import json
import pickle
from urllib.parse import urlparse
import logging
import contextlib
import time
import datetime
import fnmatch
from functools import wraps

import toml
import pandas as pd
import boto3


LOGGER = logging.getLogger(__name__)
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


if "flowCat_tmp" in os.environ:
    TMP_PATH = os.environ["flowCat_tmp"]
    LOGGER.warning("Setting tmp folder to %s", TMP_PATH)
else:
    TMP_PATH = "tmp"


# Overwrites existing data if true
if "flowCat_clobber" in os.environ:
    CLOBBER = bool(os.environ["flowCat_clobber"])
    LOGGER.warning(f"Setting clobber to {CLOBBER}")
else:
    CLOBBER = False


class Singleton(abc.ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FileBackend(metaclass=Singleton):

    @abc.abstractmethod
    def get(self, localpath, netloc, path):
        """Get data from remote location."""
        pass

    @abc.abstractmethod
    def put(self, localpath, netloc, path):
        """Put data into remote location."""
        pass

    @abc.abstractmethod
    def exists(self, netloc, path):
        """Check whether the given path exists."""
        pass

    @abc.abstractmethod
    def ls(self, netloc, path, files_only=False, delimiter="/"):
        pass


class S3Backend(FileBackend):

    continuation_token = "list_even_more"

    def __init__(self):
        self.client = boto3.client("s3")

    def extend(self, path, *args):
        """Concatenate given elements and return a string representation."""
        if not path.endswith("/"):
            path += "/"
        return path + "/".join(str(a) for a in args)

    def get(self, localpath, netloc, path):
        path = str(path).lstrip("/")

        if not localpath.exists():
            LOGGER.debug("%s download", path)
            localpath.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(netloc, path, str(localpath))
        return localpath

    def put(self, localpath, netloc, path, clobber=False):
        """
        Args:
            localpath: Temp filepath containing locally written file.
            netloc: Hostname for remote resource
            path: path on remote resource
            clobber: If true will overwrite remote existing data
        """
        path = str(path).lstrip("/")

        if not clobber:
            assert not self.exists(netloc, path)
        localpath.parent.mkdir(parents=True, exist_ok=True)
        self.client.upload_file(str(localpath), netloc, str(path))

    def exists(self, netloc, path):
        """Check existence in S3 by head-request for an object."""
        path = str(path).lstrip("/")

        return bool(self.ls(netloc, path, files_only=False))

    def ls(self, netloc, path, files_only=False, delimiter="/"):
        path = str(path).lstrip("/")

        files = []
        prefixes = []
        rargs = {
            "Bucket": netloc,
            "Prefix": path,
        }
        if delimiter:
            rargs["Delimiter"] = delimiter
        while True:
            resp = self.client.list_objects_v2(**rargs)
            if "Contents" in resp:
                files += [c["Key"] for c in resp["Contents"]]
            if "CommonPrefixes" in resp:
                prefixes += [c["Prefix"] for c in resp["CommonPrefixes"]]
            if resp["IsTruncated"]:
                rargs["ContinuationToken"] = resp["NextContinuationToken"]
            else:
                break

        if len(prefixes) == 1 and not files:
            return self.ls(
                netloc, path + "/", files_only=files_only, delimiter=delimiter)
        return [p.replace(path, "") for p in (files + prefixes)]

    def glob(self, netloc, path, pattern):
        if not pattern.startswith("/"):
            delim = "/"
        else:
            delim = None

        if not path.endswith("/"):
            path += "/"
        all_files = self.ls(netloc, path, delimiter=delim)
        matched = [f for f in all_files if fnmatch.fnmatch(f, pattern)]
        return matched


class LocalBackend(FileBackend):

    def extend(self, path, *args):
        """Concatenate given elements and return a string representation."""
        return str(pathlib.PurePath(path, *args))

    def get(self, localpath, netloc, path):
        if localpath.exists():
            return localpath
        else:
            raise RuntimeError(f"File not found {localpath}")

    def put(self, localpath, netloc, path, clobber=False):
        """Local backend will never write anywhere, since localpath and path are the same."""
        pass

    def exists(self, netloc, path):
        """Directly check whether the path exists on the local system."""
        return pathlib.Path(path).exists()

    def ls(self, netloc, path, files_only=False, delimiter="/"):
        files = [f for f in pathlib.Path(path).glob("*")]
        if files_only:
            files = [f for f in files if f.is_file()]
        return files

    def glob(self, netloc, path, pattern):
        return pathlib.Path(path).glob(pattern)


def get_backend(scheme):
    if not scheme:
        backend = LocalBackend()
    elif scheme == "s3":
        backend = S3Backend()
    else:
        raise TypeError(f"Unknown scheme {self.scheme}")
    return backend


class URLPath:
    """Combines url and pathlib.
    Manages two representations, one remote and one local. If the given
    path is local, both will be equivalent.
    """
    __slots__ = ("scheme", "netloc", "path", "_local", "_backend")

    def __init__(self, path, *args):
        if isinstance(path, URLPath):
            self.scheme = path.scheme
            self.netloc = path.netloc
            self.path = path.path
        else:
            urlpath = urlparse(str(path))
            self.scheme = urlpath.scheme
            self.netloc = urlpath.netloc
            self.path = urlpath.path

        self._backend = get_backend(self.scheme)

        # add args
        if args:
            self.path = self._backend.extend(self.path, *args)

        self._local = None

    @property
    def local(self):
        if self._local is None:
            if self.scheme:
                self._local = pathlib.Path(TMP_PATH, self.netloc + self.path)
            else:
                self._local = pathlib.Path(self.path)
        return self._local

    @property
    def remote(self):
        return bool(self.scheme)

    def exists(self):
        """Get if the given resource already exists."""
        return self._backend.exists(self.netloc, self.path)

    def get(self):
        """Load the file if it is not already local."""
        return self._backend.get(self.local, self.netloc, self.path)

    def put(self, writefun):
        """Create a new file using a given write function.
        Args:
            writefun: Function writing data to a given path.
        """
        if self.local.exists() and not CLOBBER:
            raise RuntimeError(f"{self} already exists and clobber is {CLOBBER}. Use env var flowCat_clobber to overwrite files.")
        self.local.parent.mkdir(parents=True, exist_ok=True)
        writefun(str(self.local))
        self._backend.put(self.local, self.netloc, self.path, clobber=CLOBBER)

    def clear(self):
        """Remove temporary file if remote resource."""
        if self.scheme and self.local.exists():
            LOGGER.debug("%s removed", self._local)
            self._local.unlink()

    def ls(self, **kwargs):
        return [self.__class__(self, p) for p in self._backend.ls(self.netloc, self.path, **kwargs)]

    def glob(self, pattern):
        return [self.__class__(self, p) for p in self._backend.glob(self.netloc, self.path, pattern)]

    def __truediv__(self, other):
        """Append to the path as another level."""
        return self.__class__(self, other)

    def __add__(self, other):
        """Addition will simply concatenate the fragment to the current
        string representation."""
        return self.__class__(str(self) + str(other))

    def __radd__(self, other):
        """Implement concatenation also if self is in second place."""
        return self.__class__(str(other) + str(self))

    def __repr__(self):
        if self.scheme:
            return f"{self.scheme}://{self.netloc}{self.path}"
        return f"{self.path}"

    def __lt__(self, other):
        return str(self) < str(other)

    def __getstate__(self):
        """Override default pickling behaviour of getting the dict."""
        return (self.scheme, self.netloc, self.path)

    def __setstate__(self, state):
        """Retore instance attributes."""
        self.scheme, self.netloc, self.path = state
        self._backend = get_backend(self.scheme)


def get_urlpath(fun):
    @wraps(fun)
    def get_local(path, *args, **kwargs):
        if isinstance(path, URLPath):
            path = path.get()
        return fun(path, *args, **kwargs)
    return get_local


def put_urlpath(fun):
    @wraps(fun)
    def put_local(data, path):
        if isinstance(path, URLPath):
            return path.put(lambda p: fun(data, p))
        if not CLOBBER and pathlib.Path(path).exists():
            raise RuntimeError(f"{path} already exists.")
        return fun(data, path)

    return put_local


@get_urlpath
def load_json(path):
    """Load json data from a path as a simple function."""
    with open(str(path), "r") as jspath:
        data = json.load(jspath)
    return data


@put_urlpath
def save_json(data, path):
    """Write json data to a file as a simple function."""
    with open(str(path), "w") as jsfile:
        json.dump(data, jsfile)


@get_urlpath
def load_pickle(path):
    with open(str(path), "rb") as pfile:
        data = pickle.load(pfile)
    return data


@put_urlpath
def save_pickle(data, path):
    """Write data to the given path as a pickle."""
    with open(str(path), "wb") as pfile:
        pickle.dump(data, pfile)


@get_urlpath
def load_toml(path):
    with open(str(path), "r") as f:
        data = toml.load(f)
    return data


@put_urlpath
def save_toml(data, path):
    with open(str(path), "w") as f:
        toml.dump(data, f)


def to_json(data):
    return json.dumps(data, indent=4)


def to_toml(data):
    return toml.dumps(data)


@get_urlpath
def load_csv(path):
    data = pd.read_csv(str(path), index_col=0)
    return data


@put_urlpath
def save_csv(data, path):
    data.to_csv(str(path))


def load_file(path):
    """Automatically infer format from filepath ending."""
    if str(path).endswith(".json"):
        return load_json(path)
    if str(path).endswith(".p"):
        return load_pickle(path)
    if str(path).endswith(".toml"):
        return load_toml(path)
    raise TypeError(f"Unknown suffix in {path}")


def save_file(data, path, *args, **kwargs):
    """Automatically infer save format from filepath ending."""
    if str(path).endswith(".json"):
        return save_json(data, path, *args, **kwargs)
    if str(path).endswith(".p"):
        return save_pickle(data, path, *args, **kwargs)
    if str(path).endswith(".toml"):
        return save_toml(data, path, *args, **kwargs)
    raise TypeError(f"Unknown suffix in {path}")


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
        count.set_index(["label", "group"], inplace=True)
        count = count.loc[~count.index.duplicated(keep='first')]
        if counts is None:
            counts = count
        else:
            counts = counts.add(count, fill_value=0)
    counts = counts / len(tubes)
    return counts
