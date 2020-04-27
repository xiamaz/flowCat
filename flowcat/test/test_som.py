import unittest

import numpy as np
from numpy.testing import assert_array_equal

from flowcat.types import som


class TestSOM(unittest.TestCase):

    def test_init(self):
        testsom = som.SOM(np.array([
            [[3, 4], [1, 2],],
            [[5, 6], [7, 8],],
        ]), ["A", "B"])
        self.assertEqual(testsom.shape, (2, 2, 2))
        self.assertEqual(testsom.markers, ["A", "B"])

    def test_indexing(self):
        testsom = som.SOM(np.array([
            [[3, 4], [1, 2],],
            [[5, 6], [7, 8],],
        ]), ["A", "B"])
        selection = testsom[["A"]]
        assert_array_equal(selection.data, np.array([
            [[3,], [1,]],
            [[5,], [7,]],
        ]))
        selection = testsom[:, :, ["B"]]
        assert_array_equal(selection.data, np.array([
            [[4,], [2,]],
            [[6,], [8,]],
        ]))
        selection = testsom[:, :, ["B", "A"]]
        assert_array_equal(selection.data, np.array([
            [[4, 3], [2, 1]],
            [[6, 5], [8, 7]],
        ]))
        self.assertEqual(selection.markers, ["B", "A"])
