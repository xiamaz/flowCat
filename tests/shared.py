"""
Shared functions used by all tests.
"""
import os
import pathlib
import numpy as np
import pandas as pd
import flowcat.utils as utils

# Path to test directory containing test files
TESTPATH = pathlib.Path(__file__).parent
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
