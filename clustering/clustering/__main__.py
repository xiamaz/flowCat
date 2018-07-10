import os
import logging

from .cmd_args import get_args
from .clustering import Clustering


def configure_print_logging(rootname="clustering"):
    """Configure default logging for visual output to stdout."""
    rootlogger = logging.getLogger(rootname)
    rootlogger.setLevel(logging.INFO)
    formatter = logging.Formatter()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    rootlogger.addHandler(handler)


configure_print_logging()
clustering = Clustering.from_args(get_args())
clustering.run()
