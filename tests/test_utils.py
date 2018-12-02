import unittest
import pickle

from .shared import *


class TestURLPathBase(unittest.TestCase):

    name = "testpath"
    netloc = ""
    scheme = ""
    path = "testpath"
    local = "testpath"
    remote = False
    exists = False

    def setUp(self):
        self.url = utils.URLPath(self.name)

    def test_creation(self):
        self.assertEqual(self.url.netloc, self.netloc)
        self.assertEqual(self.url.scheme, self.scheme)
        self.assertEqual(self.url.path, self.path)
        self.assertEqual(str(self.url.local), self.local)
        self.assertEqual(self.url.exists(), self.exists)
        self.assertEqual(self.url.remote, self.remote)

    def test_repr(self):
        self.assertEqual(repr(self.url), self.name)


class TestLocalPath(TestURLPathBase):
    name = str(DATAPATH)
    netloc = ""
    scheme = ""
    path = str(DATAPATH)
    local = str(DATAPATH)
    remote = False
    exists = True

    def test_ls(self):
        self.assertListEqual(
            [str(l) for l in self.url.ls()],
            [str(l) for l in DATAPATH.glob("*")],
        )


class TestLocalPathNonExist(TestURLPathBase):
    name = "/tmp/flowcat_nothing"
    netloc = ""
    scheme = ""
    path = "/tmp/flowcat_nothing"
    local = "/tmp/flowcat_nothing"
    remote = False
    exists = False


class TestPathSerialization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = utils.URLPath("s3://serialize.me/now")

    def test_pickling(self):
        """Test that pickling works correctly."""
        pickled = pickle.dumps(self.path)
        unpickled = pickle.loads(pickled)
        with self.subTest("Same string result"):
            self.assertEqual(str(self.path), str(unpickled))
