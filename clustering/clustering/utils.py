"""Currently only supports S3 objects."""
import os
import pathlib
from urllib.parse import urlparse
import logging
import datetime
import boto3

def resolve_s3(path, temp):
    if temp is None:
        raise RuntimeError(
            "No temp directorty specified. Cannot download from remote resource."
        )
    s3url = urlparse(path)
    s3file = s3url.path.lstrip("/")
    os.makedirs(temp, exist_ok=True)
    dest = pathlib.PurePath(temp, s3file)
    boto3.client("s3").download_file(
        s3url.netloc, s3file, str(dest)
    )
    print(os.listdir(temp))
    return str(dest)

def upload_s3(dest, writefun, temp):
    s3url = urlparse(dest)
    filepath = s3url.path.lstrip("/")

    tempdest = pathlib.PurePath(temp, filepath)
    os.makedirs(str(tempdest.parent), exist_ok=True)
    writefun(str(tempdest))
    boto3.client("s3").upload_file(str(tempdest), s3url.netloc, filepath)

def get_file_path(path, temp=None):
    """Resolve and download remote files to cache.
    Directly resolve local files."""
    if path.lower().startswith("s3://"):
        return resolve_s3(path, temp)
    return path

def put_file_path(path, writefun, temp):
    """Optionally upload file to remote location."""
    if path.lower().startswith("s3://"):
        return upload_s3(path, writefun, temp)
    writefun(path)
    return path

def create_stamp():
    """Create timestamp usable for filepaths"""
    stamp = datetime.datetime.now()
    return stamp.strftime("%Y%m%d_%H%M")


def configure_print_logging(rootname="clustering"):
    rootlogger = logging.getLogger(rootname)
    rootlogger.setLevel(logging.INFO)
    formatter = logging.Formatter()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    rootlogger.addHandler(handler)
