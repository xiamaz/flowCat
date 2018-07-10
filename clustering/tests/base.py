import os
import unittest


from clustering.case_collection import CaseCollection


def resolve_path(path):
    """Get absolute path relative to the location of the test program."""
    # get base path of the test folder
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, path)


class BaseCollection(unittest.TestCase):
    """Create basic assortment of utilities for testing."""

    case_json_path = resolve_path("data/case_info.json")
    tmppath = resolve_path("data/fcs")

    def setUp(self):
        self.collection = CaseCollection(self.case_json_path)


class BaseView(BaseCollection):
    """Basic operations on views."""
    def setUp(self):
        super().setUp()
        self.view = self.collection.create_view(
            bucketname="mll-flowdata", tmpdir=self.tmppath
        )
