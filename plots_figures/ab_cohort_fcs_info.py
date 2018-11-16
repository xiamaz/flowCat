"""Basic information on FCS files."""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent))

from flowcat.data import case
