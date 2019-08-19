import logging

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


def create_handler(handler, fmt=LOGGING_FORMAT, level=logging.DEBUG):
    """Add a logging formatter to the given Handler."""
    handler.setLevel(level)
    if not isinstance(fmt, logging.Formatter):
        fmt = logging.Formatter(fmt)
    handler.setFormatter(fmt)
    return handler
