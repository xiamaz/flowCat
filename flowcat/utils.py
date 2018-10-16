"""Currently only supports S3 objects."""
import abc
import os
import pathlib
import json
import pickle
from urllib.parse import urlparse
import logging
import datetime
import fnmatch
from functools import wraps

import toml
import pandas as pd
import boto3
from botocore.exceptions import ClientError


TMP_PATH = "tmp"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


LOGGER = logging.getLogger(__name__)


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
    def ls(self, netloc, path, files_only=False):
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

        return files + prefixes

    def glob(self, netloc, path, pattern):
        all_files = self.ls(netloc, path, delimiter=None)
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

    def ls(self, netloc, path, files_only=False):
        files = [f for f in pathlib.Path(path).glob("*")]
        if files_only:
            files = [f for f in files if f.is_file()]
        return files

    def glob(self, netloc, path, pattern):
        return pathlib.Path(path).glob(pattern)


class URLPath(object):
    """Combines url and pathlib.
    Manages two representations, one remote and one local. If the given
    path is local, both will be equivalent.
    """
    __slots__ = ("scheme", "netloc", "path", "_local", "_backend")

    def __init__(self, path, *args):
        urlpath = urlparse(str(path))
        self.scheme = urlpath.scheme
        self.netloc = urlpath.netloc
        self.path = urlpath.path

        if not self.scheme:
            self._backend = LocalBackend()
        elif self.scheme == "s3":
            self._backend = S3Backend()
        else:
            raise TypeError(f"Unknown scheme {self.scheme}")

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

    def put(self, writefun, clobber=False):
        """Create a new file using a given write function.
        Args:
            writefun: Function writing data to a given path.
            clobber: Overwrite data in the location.
        """
        if not self.local.exists() or clobber:
            self.local.parent.mkdir(parents=True, exist_ok=True)
            writefun(str(self.local))

        self._backend.put(self.local, self.netloc, self.path, clobber=clobber)

    def clear(self):
        """Remove temporary file if remote resource."""
        if self.scheme and self.local.exists():
            LOGGER.debug("%s removed", self._local)
            self._local.unlink()

    def ls(self):
        return [self.__class__(self, p) for p in self._backend.ls(self.netloc, self.path)]

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


def get_urlpath(fun):
    @wraps(fun)
    def get_local(path, *args, **kwargs):
        if isinstance(path, URLPath):
            path = path.get()
        return fun(path, *args, **kwargs)
    return fun


def put_urlpath(fun):
    @wraps(fun)
    def put_local(data, path, clobber=False):
        if isinstance(path, URLPath):
            return path.put(lambda p: fun(data, p), clobber=clobber)
        if clobber and pathlib.Path(path).exists():
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
