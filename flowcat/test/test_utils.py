# pylint: skip-file
# flake8: noqa
import unittest
import pickle

from flowcat import utils


class TestURLPath(unittest.TestCase):
    def test_concatenation(self):
        cases = [
            ("a", "b", "a/b"),
            ("/c", "d", "/c/d"),
            ("file:///a", "telnet", "file:///a/telnet")
        ]
        for part_a, part_b, expected in cases:
            url_a = utils.URLPath(part_a)
            url_b = utils.URLPath(part_b)
            result = url_a / url_b
            self.assertEqual(str(result), expected)

    def test_urls(self):
        cases = [
            ("a", "", ""),
            ("https://a", "https", "a"),
            ("https://dest.de/a", "https", "dest.de"),
        ]
        for url, scheme, netloc in cases:
            result = utils.URLPath(url)
            self.assertEqual(result._scheme, scheme)
            self.assertEqual(result._netloc, netloc)

    def test_addition(self):
        cases = [
            ("testfile", "as", "testfileas"),
            ("/a/", "test", "/atest"),  # trailing slashes will get removed on creation
            ("/file", ".lmd", "/file.lmd"),
        ]
        for part_a, part_b, expected in cases:
            result = utils.URLPath(part_a) + part_b
            self.assertEqual(str(result), expected)
