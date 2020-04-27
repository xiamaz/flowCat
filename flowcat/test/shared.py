"""
Shared functions used by all tests.
"""
import unittest

from numpy.testing import assert_array_equal
import pandas as pd

from flowcat.types import fcsdata as fcs


class FlowcatTestCase(unittest.TestCase):
    def assert_fcs_equal(self, data1: fcs.FCSData, data2: fcs.FCSData):
        """Ensure that data and column information in two fcs are the same."""
        assert_array_equal(data1.data, data2.data)
        self.assertListEqual(data1.channels, data2.channels)
