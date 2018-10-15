import unittest
import pathlib

from flowcat import utils


TESTPATH = pathlib.Path(__file__).parent
DATAPATH = TESTPATH / "data"
utils.TMP_PATH = "tmp_test"


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


class TestRemotePath(TestURLPathBase):
    name = "s3://mll-flowdata"
    netloc = "mll-flowdata"
    scheme = "s3"
    path = ""
    local = "tmp_test/mll-flowdata"
    remote = True
    exists = True

    def test_ls(self):
        self.assertEqual(
            [str(l) for l in self.url.ls()],
            [
                f"{self.name}/case_info.json",
                f"{self.name}/CLL-9F/",
                f"{self.name}/meta/",
                f"{self.name}/origdata/"]
        )


class TestRemoteObject(TestURLPathBase):
    name = "s3://mll-flowdata/CLL-9F/case_info.json"
    netloc = "mll-flowdata"
    scheme = "s3"
    path = "/CLL-9F/case_info.json"
    local = "tmp_test/mll-flowdata/CLL-9F/case_info.json"
    remote = True
    exists = True


class TestRemoteObjectNonExist(TestURLPathBase):
    name = "s3://mll-flowdata/CLL-9F/whatever_that_doesnt_exist"
    netloc = "mll-flowdata"
    scheme = "s3"
    path = "/CLL-9F/whatever_that_doesnt_exist"
    local = "tmp_test/mll-flowdata/CLL-9F/whatever_that_doesnt_exist"
    remote = True
    exists = False


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
