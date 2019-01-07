import unittest

from .shared import *
from .test_utils import TestURLPathBase


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
            [str(l) for l in (self.url / "histogram/").ls()],
            [
                f"{self.name}/histogram/abstract_normal/",
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
