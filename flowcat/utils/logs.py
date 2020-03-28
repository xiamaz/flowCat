"""
Python logging flow: logger -> handlers

Log level can be set at logger level and at handler level. So logging with
different levels to stream and file are for example possible.
"""

import logging
from logging import FileHandler, StreamHandler

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def add_logger(log, handlers, level=logging.DEBUG):
    """Add logger or name of a logger to a list of handlers."""
    if isinstance(log, str):
        log = logging.getLogger(log)
    elif not isinstance(log, logging.Logger):
        raise TypeError("Wrong type for log")

    log.setLevel(level)
    for handler in handlers:
        log.addHandler(handler)


def create_handler(handler, fmt=LOGGING_FORMAT, level=logging.DEBUG) -> logging.Handler:
    """Add a logging formatter to the given Handler."""
    handler.setLevel(level)
    if not isinstance(fmt, logging.Formatter):
        fmt = logging.Formatter(fmt)
    handler.setFormatter(fmt)
    return handler


def print_stream() -> StreamHandler:
    """Create a stream to stdout."""
    return StreamHandler()


def create_logging_handlers(logging_path: str) -> [logging.Handler]:
    """Create logging to both file and stderr."""
    loggers = [
        create_handler(logging.FileHandler(logging_path)),
        create_handler(print_stream()),
    ]
    return loggers


def setup_logging(logging_path: "URLPath", name: str, fc_level=logging.INFO) -> logging.Logger:
    """Create a single logger object. To be used in main scripts."""
    logging_path.parent.mkdir()

    logger = logging.getLogger(name)

    fc_logger = logging.getLogger("flowcat")
    handlers = create_logging_handlers(logging_path)
    add_logger(logger, handlers)
    add_logger(fc_logger, handlers, level=fc_level)
    return logger
