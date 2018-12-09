import unittest
import shutil
from .shared import *
from flowcat import configuration


class TestConfigIO(unittest.TestCase):

    def setUp(self):

        self.tmp_folder = TESTPATH / "tmp"
        self.tmp_folder.mkdir()

    def test_writing_reading(self):
        class Simple(configuration.Config):
            _schema = {
                "a": ((int,), None),
            }

        orig = Simple({"a": 10})
        path = self.tmp_folder / "sample.toml"
        orig.to_file(path)

        new = Simple.from_file(path)
        self.assertEqual(orig, new)

    def tearDown(self):
        shutil.rmtree(self.tmp_folder)


class TestPathConfigIO(TestConfigIO):

    def test_writing_reading(self):
        orig = configuration.PathConfig({})
        path = self.tmp_folder / "pathsample.toml"
        orig.to_file(path)

        new = configuration.PathConfig.from_file(path)
        self.assertEqual(orig, new)


class TestSOMConfigIO(TestConfigIO):

    def test_writing_reading(self):
        orig = configuration.SOMConfig({"name": "test", "tfsom": {"model_name": "test"}})
        path = self.tmp_folder / "somsample.toml"
        orig.to_file(path)

        new = configuration.SOMConfig.from_file(path)
        self.assertEqual(orig, new)
