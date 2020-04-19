import unittest

import numpy as np
from flowcat.utils import URLPath
from flowcat.dataset import fcs
from flowcat.preprocessing import edge_removal
from flowcat.types.marker import Marker

from .shared import FlowcatTestCase


def create_fcs(data, meta):
    array = np.array(data)
    mask = np.ones(shape=array.shape)
    return fcs.FCSData(
        (array, mask),
        channels=[Marker.name_to_marker(n, m) for n, m in meta.items()])


class EdgeRemovalTestCase(FlowcatTestCase):
    def test_simple(self):
        data = create_fcs(
            [
                [1, 2, 3, 4],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [10, 10, 0, 10],
                [5, 6, 12, 10],
                [4, 2, 1, 9],
            ],
            meta={
                "a": fcs.ChannelMeta(0, 10, (0, 0), 0),
                "b": fcs.ChannelMeta(0, 10, (0, 0), 0),
                "c": fcs.ChannelMeta(0, 12, (0, 0), 0),
                "d": fcs.ChannelMeta(0, 10, (0, 0), 0),
            }
        )

        expected = create_fcs(
            [
                [1, 2, 3, 4],
                [1, 1, 1, 1],
                [5, 6, 12, 10],
                [4, 2, 1, 9],
            ],
            meta={
                "a": fcs.ChannelMeta(0, 10, (0, 0), 0),
                "b": fcs.ChannelMeta(0, 10, (0, 0), 0),
                "c": fcs.ChannelMeta(0, 12, (0, 0), 0),
                "d": fcs.ChannelMeta(0, 10, (0, 0), 0),
            }
        )

        edge_data = create_fcs(
            [
                [0],
                [10],
            ],
            meta={"a": None},
        )
        model = edge_removal.EdgeEventFilter(["a"])
        model.fit(edge_data)
        result = model.transform(data)

        self.assert_fcs_equal(result, expected)

    def test_fit_filter(self):
        data = create_fcs(
            [
                [1, 2, 3, 4],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [10, 10, 0, 10],
                [5, 6, 12, 10],
                [4, 2, 1, 9],
                [2, 10, 0, 10],
            ],
            meta={
                "a": fcs.ChannelMeta(0, 10, (0, 0), 0),
                "b": fcs.ChannelMeta(0, 10, (0, 0), 0),
                "c": fcs.ChannelMeta(0, 12, (0, 0), 0),
                "d": fcs.ChannelMeta(0, 10, (0, 0), 0),
            }
        )

        expected = create_fcs(
            [
                [1, 2, 3, 4],
                [1, 1, 1, 1],
                [5, 6, 12, 10],
                [4, 2, 1, 9],
            ],
            meta={
                "a": fcs.ChannelMeta(0, 10, (0, 0), 0),
                "b": fcs.ChannelMeta(0, 10, (0, 0), 0),
                "c": fcs.ChannelMeta(0, 12, (0, 0), 0),
                "d": fcs.ChannelMeta(0, 10, (0, 0), 0),
            }
        )
        model = edge_removal.EdgeEventFilter(["a", "b"])
        result = model.fit_transform(data)
        self.assert_fcs_equal(result, expected)
