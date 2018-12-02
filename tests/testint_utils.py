import unittest

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


class TestRemotePath(TestURLPathBase):
    name = "s3://flowcat-test"
    netloc = "flowcat-test"
    scheme = "s3"
    path = ""
    local = "tmp_test/flowcat-test"
    remote = True
    exists = True

    def test_ls(self):
        self.assertEqual(
            [str(l) for l in self.url.ls()],
            [
                f"{self.name}/test_cases.json",
                f"{self.name}/histogram/",
                f"{self.name}/small_dataset/",
                f"{self.name}/som/",
            ]
        )

    def test_glob(self):
        url = self.url / "small_dataset/"
        results = url.glob("3 CLL 9F 01 N06 001")
        self.assertTrue(all("3 CLL 9F 01 N06 001" in str(r) for r in results))


class TestRemoteObject(TestURLPathBase):
    name = "s3://flowcat-test/small_dataset/case_info.json"
    netloc = "flowcat-test"
    scheme = "s3"
    path = "/small_dataset/case_info.json"
    local = "tmp_test/flowcat-test/small_dataset/case_info.json"
    remote = True
    exists = True


class TestRemoteObjectNonExist(TestURLPathBase):
    name = "s3://flowcat-test/small_dataset/whatever_that_doesnt_exist"
    netloc = "flowcat-test"
    scheme = "s3"
    path = "/small_dataset/whatever_that_doesnt_exist"
    local = "tmp_test/flowcat-test/small_dataset/whatever_that_doesnt_exist"
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
