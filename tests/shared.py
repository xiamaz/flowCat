"""
Shared functions used by all tests.
"""
import os
import pathlib
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import flowcat.utils as utils
from flowcat.dataset import fcs

# Path to test directory containing test files
TESTPATH = utils.URLPath(__file__).parent
# Path to additional data needed for some tests
DATAPATH = TESTPATH / "data"
# Path to fcs files and metadata directory
FCS_PATH = DATAPATH / "small_dataset"
# Path to set of self-organizing maps
SOM_PATH = DATAPATH / "som" / "testrun_s32_ttoroid"
# Path to set of histograms
HISTO_PATH = DATAPATH / "histogram" / "abstract_normal"

# set utils tmp path to another location
utils.TMP_PATH = "tmp_test"


def create_fcs(data, columns, meta=None) -> fcs.FCSData:
    """Create fcs data from literal information."""
    if meta:
        meta = create_fcs_meta(meta)
    return fcs.FCSData(pd.DataFrame(data, columns=columns), meta=meta)


def create_fcs_meta(meta_dict: dict) -> dict:
    """Create dict of namedtuples."""
    return {
        name: fcs.ChannelMeta(
            pd.Interval(*interval),
            exists
        )
        for name, (interval, exists) in meta_dict.items()
    }


class FlowcatTestCase(unittest.TestCase):
    def assert_fcs_equal(self, data1: fcs.FCSData, data2: fcs.FCSData):
        """Ensure that data and column information in two fcs are the same."""
        assert_frame_equal(data1.data, data2.data)

        self.assertDictEqual(data1.meta, data2.meta)


class MockLoader:
    """Return mock data for different path calls."""

    URLPath = utils.URLPath

    def load_csv(self, path):
        path = str(path)
        if path == "som9":
            # get a regular matrix from 0 to 89 in 9 rows with 10 values each
            data = pd.DataFrame(np.arange(90).reshape(9, 10))
        elif path == "som4":
            data = pd.DataFrame(np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]))
        elif path == "somcount4":
            data = pd.DataFrame(np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]))
            data["counts"] = np.ones(4) * 90
        elif path == "sombothcount4":
            data = pd.DataFrame(np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]))
            data["counts"] = np.ones(4) * 90
            data["count_prev"] = np.ones(4) * 100
        elif path == "somprevcount4":
            data = pd.DataFrame(np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]))
            data["count_prev"] = np.ones(4) * 100
        else:
            raise KeyError(f"No mock data available for {path}")
        return data


def get_test_dataset(name):
    """Return test dataset with the given name"""
    return DATAPATH / name
