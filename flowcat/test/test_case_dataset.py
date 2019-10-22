"""
Case collection tests.
"""
import unittest
import json
import pathlib
import datetime

import pandas as pd
from pandas.testing import assert_series_equal

from flowcat import mappings
from flowcat.utils.time_timers import str_to_date
from flowcat.dataset import case_dataset, case, sample


def create_case(id: str, date="2011-11-11", used_material="PB", samples=None, **kwargs):
    used_material = mappings.Material.from_str(used_material)
    date = str_to_date(date)
    if samples:
        samples = [sample.Sample(**s) for s in samples]
    else:
        samples = []
    return case.Case(
        id=id,
        date=date,
        used_material=used_material,
        samples=samples,
        **kwargs
    )


def create_case_dataset(case_args):
    cases = [create_case(**c) for c in case_args]
    return case_dataset.CaseCollection(cases=cases)


class TestCaseCollection(unittest.TestCase):
    def test_basic(self):
        case_args = [
            {
                "id": "1",
                "samples": [
                    {"id": "a", "case_id": "1", "date": datetime.date(2011, 11, 11), "tube": "1"},
                    {"id": "b", "case_id": "1", "date": datetime.date(2011, 11, 11), "tube": "2"}
                ]
            }
        ]
        dataset = create_case_dataset(case_args)
        self.assertEqual(len(dataset), len(case_args))
        self.assertEqual(dataset.tubes, ["1", "2"])
        self.assertEqual(dataset.group_count, {None: 1})
        self.assertEqual(dataset.groups, [None])
        self.assertEqual(dataset.labels, ["1"])

    def test_sampling(self):
        case_args = [
            {
                "id": "1",
                "group": "a",
            },
            {
                "id": "2",
                "group": "b",
            },
            {
                "id": "3",
                "group": "c",
            },
            {
                "id": "4",
                "group": "c",
            },
        ]
        dataset = create_case_dataset(case_args)
        self.assertEqual(dataset.group_count, {"a": 1, "b": 1, "c": 2})
        cases = [
            ((1, ["b", "c"]), {"b": 1, "c": 1}),
            ((10000, ["b", "c"]), {"b": 1, "c": 2}),
            ((2, ["a"]), {"a": 1}),
            ((2, ["a", "b"]), {"a": 1, "b": 1}),
            ((2, ["a", "b", "c"]), {"a": 1, "b": 1, "c": 2}),
        ]
        for args, expected in cases:
            sampled = dataset.sample(*args)
            self.assertDictEqual(sampled.group_count, expected)

    def test_balancing(self):
        case_args = [
            {
                "id": "1",
                "group": "a",
            },
            {
                "id": "2",
                "group": "b",
            },
            {
                "id": "3",
                "group": "c",
            },
        ]
        dataset = create_case_dataset(case_args)
        self.assertEqual(dataset.group_count, {"a": 1, "b": 1, "c": 1})
        cases = [
            ((2,), {"a": 2, "b": 2, "c": 2}),
            ((100,), {"a": 100, "b": 100, "c": 100}),
        ]
        for args, expected in cases:
            sampled = dataset.balance(*args)
            self.assertEqual(sampled.group_count, expected)

        cases = [
            ({"a": 100, "b": 1, "c": 1}, {"a": 100, "b": 1, "c": 1}),
            ({"a": 100, "b": 1, "c": 0}, {"a": 100, "b": 1}),
        ]
        for arg, expected in cases:
            sampled = dataset.balance_per_group(arg)
            self.assertEqual(sampled.group_count, expected)

    def test_split(self):
        cases = [
            ([
                {"id": "1", "group": "a"},
                {"id": "2", "group": "a"},
                {"id": "a1", "group": "b"},
                {"id": "a2", "group": "b"},
            ], [
                ((0.5, True), ({"a": 1, "b": 1}, {"a": 1, "b": 1})),
                ((0.8, True), ({"a": 2, "b": 2}, {})),
                ((0.2, True), ({}, {"a": 2, "b": 2})),
            ]),
            ([
                {"id": "1", "group": "a"},
                {"id": "2", "group": "a"},
                {"id": "c1", "group": "a"},
                {"id": "c2", "group": "a"},
                {"id": "a1", "group": "b"},
                {"id": "a2", "group": "b"},
                {"id": "b1", "group": "b"},
                {"id": "b2", "group": "b"},
            ], [
                ((0.5, True), ({"a": 2, "b": 2}, {"a": 2, "b": 2})),
                ((0.75, True), ({"a": 3, "b": 3}, {"a": 1, "b": 1})),
            ]),
            ([
                {"id": "1", "group": "a"},
                {"id": "2", "group": "a"},
                {"id": "c1", "group": "a"},
                {"id": "c2", "group": "a"},
                {"id": "a1", "group": "b"},
                {"id": "a2", "group": "b"},
                {"id": "b1", "group": "b"},
                {"id": "b2", "group": "b"},
                {"id": "d2", "group": "c"},
                {"id": "d4", "group": "c"},
            ], [
                ((0.5, True), ({"a": 2, "b": 2, "c": 1}, {"a": 2, "b": 2, "c": 1})),
                ((0.75, True), ({"a": 3, "b": 3, "c": 2}, {"a": 1, "b": 1})),
            ]),
        ]
        for case_args, splits in cases:
            dataset = create_case_dataset(case_args)
            for args, (expected_a, expected_b) in splits:
                part_a, part_b = dataset.create_split(*args)
                ids_a = set(c.id for c in part_a)
                ids_b = set(c.id for c in part_b)
                self.assertEqual(part_a.group_count, expected_a)
                self.assertEqual(part_b.group_count, expected_b)
                self.assertEqual(ids_a & ids_b, set())

    def test_filter(self):
        cases = [
            ([
                {"id": "1", "group": "a", "date": "2011-10-11"},
                {"id": "2", "group": "a", "date": "2011-12-11"},
                {"id": "a1", "group": "b"},
                {"id": "a2", "group": "b"},
            ], [
                ({"groups": ["a"]}, ["1", "2"]),
                ({"date": ("2011-11-11", None)}, ["2", "a1", "a2"]),
                ({"date": (None, "2011-11-11")}, ["1", "a1", "a2"])  # both ranges are inclusive
            ]),
        ]
        for case_args, filters in cases:
            dataset = create_case_dataset(case_args)
            for args, expected in filters:
                filtered = dataset.filter(**args)
                self.assertEqual(filtered.labels, expected)
