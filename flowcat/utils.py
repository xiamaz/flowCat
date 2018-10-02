"""Currently only supports S3 objects."""
import os
import pathlib
import json
import pickle
from urllib.parse import urlparse
import logging
import datetime
import boto3


TMP_PATH = "tmp"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


LOGGER = logging.getLogger(__name__)


def resolve_s3(path):
    """Download file from S3 into a temporary location.
    Currently removal of temporary files is not implemented.
    """
    s3url = urlparse(path)
    s3file = s3url.path.lstrip("/")

    dest = pathlib.Path(TMP_PATH, s3file)

    # create all folders as needed
    os.makedirs(str(dest.parent), exist_ok=True)

    if not dest.exists():
        LOGGER.debug("%s: Downloading", path)
        boto3.client("s3").download_file(
            s3url.netloc, s3file, str(dest)
        )
    else:
        LOGGER.debug("%s: Using existing in cache", path)

    return str(dest)


def upload_s3(dest, writefun):
    """Upload file to Amazon S3."""
    s3url = urlparse(dest)
    filepath = s3url.path.lstrip("/")

    tempdest = pathlib.PurePath(TMP_PATH, filepath)
    # create temp folders as needed
    os.makedirs(str(tempdest.parent), exist_ok=True)
    writefun(str(tempdest))
    boto3.client("s3").upload_file(str(tempdest), s3url.netloc, filepath)


def get_file_path(path):
    """Resolve and download remote files to cache.
    Directly resolve local files."""
    if path.lower().startswith("s3://"):
        return resolve_s3(path)
    return path


def put_file_path(path, writefun):
    """Optionally upload file to remote location."""
    if path.lower().startswith("s3://"):
        return upload_s3(path, writefun)
    os.makedirs(pathlib.PurePath(path).parent, exist_ok=True)
    writefun(path)
    return path


def load_json(path):
    """Load json data from a path as a simple function."""
    with open(str(path), "r") as jspath:
        data = json.load(jspath)
    return data


def save_json(data, path, clobber=False):
    """Write json data to a file as a simple function."""
    if not clobber and pathlib.Path(path).exists():
        raise RuntimeError(f"{path} already exists.")
    with open(str(path), "w") as jsfile:
        json.dump(data, jsfile)


def load_pickle(path):
    with open(str(path), "rb") as pfile:
        data = pickle.load(pfile)
    return data


def save_pickle(data, path, clobber=False):
    """Write data to the given path as a pickle."""
    if not clobber and pathlib.Path(path).exists():
        raise RuntimeError(f"{path} already exists.")
    with open(str(path), "wb") as pfile:
        pickle.dump(data, pfile)


def create_stamp():
    """Create timestamp usable for filepaths"""
    stamp = datetime.datetime.now()
    return stamp.strftime(TIMESTAMP_FORMAT)
