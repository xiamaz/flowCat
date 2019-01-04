"""
Test creation of SOM from initial raw data.
"""
import unittest
from flowcat import som
from flowcat.dataset import case_dataset

from . import shared


class TestSOMCreation(unittest.TestCase):
    """Multiple SOM creation scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.cases = case_dataset.CaseCollection.from_path(
            shared.get_test_dataset("small_dataset"))
