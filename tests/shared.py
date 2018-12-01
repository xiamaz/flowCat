"""
Shared functions used by all tests.
"""
import os
import pathlib
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
