"""Currently only supports S3 objects."""
import abc
import os
import pathlib
import json
import pickle
from urllib.parse import urlparse
import logging
import datetime
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

    @abc.abstractclassmethod
    def get(self, localpath, netloc, path):
        """Get data from remote location."""
        pass

    @abc.abstractclassmethod
    def put(self, localpath, netloc, path):
        """Put data into remote location."""
        pass

    @abc.abstractclassmethod
    def exists(self, netloc, path):
        """Check whether the given path exists."""
        pass


class S3Backend(FileBackend):

    def __init__(self):
        self.client = boto3.client("s3")

    def get(self, localpath, netloc, path):
        if not localpath.exists():
            LOGGER.debug("%s download", path)
            localpath.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(netloc, str(path), str(localpath))
        return localpath

    def put(self, localpath, netloc, path, clobber=False):
        """
        Args:
            localpath: Temp filepath containing locally written file.
            netloc: Hostname for remote resource
            path: path on remote resource
            clobber: If true will overwrite remote existing data
        """
        if not clobber:
            assert not self.exists(netloc, path)
        localpath.parent.mkdir(parents=True, exist_ok=True)
        self.client.upload_file(str(localpath), netloc, str(path))

    def exists(self, netloc, path):
        """Check existence in S3 by head-request for an object."""
        try:
            self.client.head_object(Bucket=netloc, key=str(path))
        except ClientError as error:
            if int(error.response["Error"]["Code"]) != 404:
                raise error
            return False
        return True


class LocalBackend(FileBackend):

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


class URLPath(object):
    """Combines url and pathlib."""
    __slots__ = ("scheme", "netloc", "path", "_local", "_backend")

    def __init__(self, path, *args):
        path = str(path) + "/" + "/".join(args)
        urlpath = urlparse(path)
        self.scheme = urlpath.scheme
        self.netloc = urlpath.netloc

        if not self.scheme:
            self._backend = LocalBackend()
            self.path = pathlib.PurePath(urlpath.path)
        elif self.scheme == "s3":
            self._backend = S3Backend()
            self.path = pathlib.PurePosixPath(urlpath.path.lstrip("/"))
        else:
            raise TypeError(f"Unknown scheme {self.scheme}")

        self._local = None

    @property
    def local(self):
        if self._local is None:
            if self.scheme:
                self._local = pathlib.Path(TMP_PATH, *self.path.parts[1:])
            else:
                self._local = pathlib.Path(self.path)
        return self._local

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

    def __truediv__(self, other):
        """Append to the path."""
        return self.__class__(self, other)

    def __repr__(self):
        if self.scheme:
            return f"{self.scheme}://{self.netloc}/{self.path}"
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
