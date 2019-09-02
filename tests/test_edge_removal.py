import unittest

import pandas as pd
from flowcat.utils import URLPath
from flowcat.dataset import fcs
from flowcat.preprocessing import edge_removal

from . import shared


class EdgeRemovalTestCase(shared.FlowcatTestCase):
    def test_simple(self):
        data = shared.create_fcs(
            [
                [1, 2, 3, 4],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [10, 10, 0, 10],
                [5, 6, 12, 10],
                [4, 2, 1, 9],
            ],
            columns=("a", "b", "c", "d"),
        )

        expected = shared.create_fcs(
            [
                [1, 2, 3, 4],
                [1, 1, 1, 1],
                [5, 6, 12, 10],
                [4, 2, 1, 9],
            ],
            columns=("a", "b", "c", "d"),
            meta={
                "a": ((0, 10, "neither"), True),
                "b": ((0, 10, "both"), True),
                "c": ((0, 12, "both"), True),
                "d": ((0, 10, "both"), True),
            }
        )

        model = edge_removal.EdgeEventFilter({"a": (0, 10)})
        result = model.transform(data)

        self.assert_fcs_equal(result, expected)

    def test_fit_filter(self):
        data = shared.create_fcs(
            [
                [1, 2, 3, 4],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [10, 10, 0, 10],
                [5, 6, 12, 10],
                [4, 2, 1, 9],
                [2, 10, 0, 10],
            ],
            columns=("a", "b", "c", "d"),
        )

        expected = shared.create_fcs(
            [
                [1, 2, 3, 4],
                [1, 1, 1, 1],
                [5, 6, 12, 10],
                [4, 2, 1, 9],
            ],
            columns=("a", "b", "c", "d"),
            meta={
                "a": ((0, 10, "neither"), True),
                "b": ((0, 10, "neither"), True),
                "c": ((0, 12, "both"), True),
                "d": ((0, 10, "both"), True),
            }
        )
        model = edge_removal.EdgeEventFilter({"a": (None, None), "b": (None, None)})
        result = model.fit_transform(data)
        self.assert_fcs_equal(result, expected)
