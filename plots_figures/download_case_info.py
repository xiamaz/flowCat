# pylint: skip-file
# flake8: noqa
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent))

from flowcat import utils
from flowcat.data.case_dataset import CaseCollection


parser = argparse.ArgumentParser(
    description="Download metadata from the specified repository. Always the latest case_info json will be used.")
parser.add_argument("path", help="Path to Data repository.", type=utils.URLPath)
parser.add_argument("--output", help="Output directory", default="data", type=pathlib.Path)

args = parser.parse_args()

# Set the download directory
utils.TMP_PATH = args.output

# Creating the CaseCollection object using an URL will ensure downloading of
# remote metadata objects
cases = CaseCollection.from_dir(args.path)
