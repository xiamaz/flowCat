"""Currently only supports S3 objects."""
import os
import pathlib
import json
from urllib.parse import urlparse
import logging
import datetime
import boto3


TMP_PATH = "tmp"


def resolve_s3(path):
    """Download file from S3 into a temporary location.
    Currently removal of temporary files is not implemented.
    """
    s3url = urlparse(path)
    s3file = s3url.path.lstrip("/")

    os.makedirs(TMP_PATH, exist_ok=True)

    dest = pathlib.PurePath(TMP_PATH, s3file)
    boto3.client("s3").download_file(
        s3url.netloc, s3file, str(dest)
    )

    return str(dest)


def upload_s3(dest, writefun):
    """Upload file to Amazon S3."""
    s3url = urlparse(dest)
    filepath = s3url.path.lstrip("/")

    tempdest = pathlib.PurePath(TMP_PATH, filepath)
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
    with open(path) as jspath:
        data = json.load(jspath)
    return data


def create_stamp():
    """Create timestamp usable for filepaths"""
    stamp = datetime.datetime.now()
    return stamp.strftime("%Y%m%d_%H%M")
