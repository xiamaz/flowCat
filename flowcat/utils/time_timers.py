"""
Basic functions for working with timestamps and timing functions.
"""
import time
import datetime
import contextlib
import collections

import numpy as np

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def str_to_date(strdate: str) -> datetime.date:
    return datetime.datetime.strptime(strdate, "%Y-%m-%d").date()


def create_stamp(stamp: "datetime" = None) -> str:
    """Create timestamp usable for filepaths"""
    if stamp is None:
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


def time_generator_logger(generator, rolling_len=20):
    """Time the given generator.
    Args:
        generator: Will be executed and results are passed through.
        rolling_len: Number of last values for generation of average time.

    Returns:
        Any value returned from the given generator.
    """
    circ_buffer = collections.deque(maxlen=rolling_len)
    time_a = time.time()
    for res in generator:
        time_b = time.time()
        time_d = time_b - time_a
        circ_buffer.append(time_d)
        time_rolling = np.mean(circ_buffer)
        print(f"Training time: {time_d}s Rolling avg: {time_rolling}s")
        time_a = time_b
        yield res
