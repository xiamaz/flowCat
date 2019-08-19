import unittest

from flowcat.utils import URLPath
from flowcat.dataset import fcs
from flowcat.preprocessing import edge_removal

from . import shared


class EdgeRemovalTestCase(unittest.TestCase):
    def test_simple(self):
        data = fcs.FCSData(shared.TESTPATH / URLPath("data/fcs/cll1_1.lmd"))

        model = edge_removal.EdgeEventFilter({"SS INT LIN": (10, 1000)})
        model.transform(data)
