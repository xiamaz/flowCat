import unittest

from .shared import *

from flowcat import configuration


class TestConfig(unittest.TestCase):

    def test_basic_init(self):
        """Test basic usage."""

        class Rudimentary(configuration.Config):
            _schema = {
                "a": ((int, None), None),
                "b": ((int, None), 5),
            }
        testobj = Rudimentary({"a": 1, "b": 2})
        self.assertEqual(testobj("a"), 1)
        self.assertEqual(testobj("b"), 2)

        # empty init should result in all default
        testobj = Rudimentary({})
        self.assertEqual(testobj("a"), None)
        self.assertEqual(testobj("b"), 5)

        class DictConfig(configuration.Config):
            _schema = {
                "a": (({int: int},), {1: 2}),
            }
        with self.subTest("Creating flexible dicts"):
            t = DictConfig({"a": {1: 5}})
            t = DictConfig({"a": {1: 5, 2: 10}})

    def test_schema_check(self):
        """Test schema check robustness"""
        class MoreComplex(configuration.Config):
            _schema = {
                "a": ((int, None), None),
                "b": ((int, None), 5),
                "d": {
                    "i": ((int,), 10),
                },
                "f": ((float,), None),  # required value
                "z": (([int], None), None),
            }
        with self.subTest("Wrong type"):
            with self.assertRaises(TypeError):
                testobj = MoreComplex({"a": 1.2, "b": 3, "f": 1.1})

            with self.assertRaises(TypeError):
                testobj = MoreComplex({"a": "1", "b": 3, "f": 1.1})

        with self.subTest("Foreign keys in schema"):
            with self.assertRaises(KeyError):
                testobj = MoreComplex({"a": 1, "c": 5, "f": 3.1})
            with self.assertRaises(KeyError):
                testobj = MoreComplex({"a": 1, "d": {"c": 5}, "f": 3.1})

        with self.subTest("Required values"):
            with self.assertRaises(TypeError):
                testobj = MoreComplex({})
            with self.assertRaises(TypeError):
                testobj = MoreComplex({"a": 10, "b": 5, "d": {"i": 11}})

        with self.subTest("Check lists"):
            with self.assertRaises(TypeError):
                testobj = MoreComplex({"f": 3.1, "z": [1.2]})
            with self.assertRaises(TypeError):
                testobj = MoreComplex({"f": 3.1, "z": [1, 2, 3, 1.2]})
            with self.assertRaises(TypeError):
                testobj = MoreComplex({"f": 3.1, "z": [1, 2, 3, "1.2", 3.2]})

        class DictConfig(configuration.Config):
            _schema = {
                "a": (({int: int},), {1: 2}),
            }

        with self.subTest("Check arbitrary dictionaries"):
            with self.assertRaises(TypeError):
                testobj = DictConfig({"a": {1: 1.2}})
            with self.assertRaises(TypeError):
                testobj = DictConfig({"a": {1: 1.2, 2: 1}})

    def test_get_default_value(self):
        class Nested(configuration.Config):
            _schema = {
                "a": {
                    "b": ((int,), 10),
                    "c": ((int,), 6),
                }
            }
        t = Nested({})
        self.assertEqual(t("a", "b"), 10)
        self.assertEqual(t("a", "c"), 6)
        self.assertEqual(t("a"), {"b": 10, "c": 6})

    def test_get_elements(self):
        class MoreComplex(configuration.Config):
            _schema = {
                "a": ((int, None), None),
                "b": ((int, None), 5),
                "d": {
                    "i": ((int,), 10),
                },
                "f": ((float,), None),  # required value
                "z": (([int], None), None),
            }
        with self.subTest("normal getting"):
            t = MoreComplex({"a": 20, "b": None, "f": 10.0})
            self.assertEqual(t("a"), 20)
            self.assertEqual(t("b"), None)
            self.assertEqual(t("f"), 10.0)

        with self.subTest("defaults getting"):
            t = MoreComplex({"a": 20, "f": 10.0})
            self.assertEqual(t("z"), None)
            self.assertEqual(t("b"), 5)

        with self.subTest("nested getting"):
            t = MoreComplex({"a": 20, "d": {"i": 100}, "f": 10.0})
            self.assertEqual(t("d", "i"), 100)

        with self.subTest("nested getting default"):
            t = MoreComplex({"a": 20, "f": 10.0})
            self.assertEqual(t("d", "i"), 10)

        class DictConfig(configuration.Config):
            _schema = {
                "a": (({int: int},), {1: 2}),
            }

        with self.subTest("dict getting"):
            t = DictConfig({"a": {1: 5}})
            self.assertEqual(t("a", 1), 5)

        with self.subTest("dict getting default"):
            t = DictConfig({})
            self.assertEqual(t("a", 1), 2)

    def test_eq(self):
        class MoreComplex(configuration.Config):
            _schema = {
                "a": ((int, None), None),
                "b": ((int, None), 5),
                "d": {
                    "i": ((int,), 10),
                },
                "f": ((float,), None),  # required value
                "z": (([int], None), None),
            }
        eq_a = MoreComplex({"f": 1.5})
        eq_b = MoreComplex({"f": 1.5, "z": None})
        self.assertEqual(eq_a, eq_b)

        neq_a = MoreComplex({"f": 9.9})
        neq_b = MoreComplex({"f": 10.0})
        self.assertNotEqual(neq_a, neq_b)


class TestPathConfig(unittest.TestCase):

    def test_initialization(self):
        with self.subTest("empty initialization"):
            t = configuration.PathConfig({})
        with self.subTest("values"):
            t = configuration.PathConfig({"input": {"a": ["a"], "b": ["b", "c"]}})
        with self.subTest("outschema"):
            with self.assertRaises(KeyError):
                t = configuration.PathConfig({"nonexist": None})
            with self.assertRaises(TypeError):
                t = configuration.PathConfig({"input": "hello"})

    def test_getter(self):
        t = configuration.PathConfig({})
        self.assertEqual(t("output", "classification"), "output/mll-sommaps/classification")
        self.assertEqual(t("input"), {
            'FCS': ['/data/flowcat-data/mll-flowdata', 's3://mll-flowdata'],
            'HISTO': ['s3://mll-flow-classification/clustering'],
            'SOM': ['output/mll-sommaps/sample_maps', 's3://mll-sommaps/sample_maps']
        })


class TestSOMConfig(unittest.TestCase):

    def test_initialization(self):
        # raise error with empty init, since we expect name fields to be filled
        with self.subTest("empty initialization"):
            with self.assertRaises(TypeError):
                configuration.SOMConfig({})

        with self.subTest("minimal initialization"):
            configuration.SOMConfig({"name": "test", "tfsom": {"model_name": "test"}, "dataset": {"selected_markers": {}}})

    def test_getter(self):
        t = configuration.SOMConfig({"name": "test", "tfsom": {"model_name": "test"}, "dataset": {"selected_markers": {}}})
        self.assertEqual(t("dataset", "names", "FCS"), "fixedCLL-9F")
        self.assertEqual(t("tfsom", "model_name"), "test")
        self.assertEqual(t("tfsom")["model_name"], "test")
