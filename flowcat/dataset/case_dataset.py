"""
Classes for managing collections of case and tubecase objects.
"""
from __future__ import annotations

import random
import logging
import collections
from dataclasses import dataclass, field
from typing import List, Tuple

from dataslots import with_slots
import numpy as np
import pandas as pd
from sklearn import model_selection

from flowcat import mappings, utils

from . import case as fc_case
from . import sample as fc_sample


LOGGER = logging.getLogger(__name__)


class DatasetError(Exception):
    pass


def case_collection_to_json(cases: CaseCollection) -> dict:
    return {
        "cases": cases.cases,
        "selected_markers": cases.selected_markers,
        "selected_tubes": cases.selected_tubes,
        "filterconfig": cases.filterconfig,
    }


def json_to_case_collection(jsonobj: dict) -> CaseCollection:
    cases = CaseCollection(
        jsonobj["cases"],
        selected_markers=jsonobj["selected_markers"],
        selected_tubes=jsonobj["selected_tubes"],
        filterconfig=jsonobj["filterconfig"],
    )
    return cases


@with_slots
@dataclass
class CaseCollection:
    """Iterable collection for cases. Base class."""

    cases: List[fc_case.Case]

    data_path: utils.URLPath = None
    meta_path: utils.URLPath = None

    selected_markers: dict = None
    selected_tubes: List[str] = None
    filterconfig: list = field(default_factory=list)

    @property
    def tubes(self):
        """Get list of unique tubes in dataset.
        Returns:
            A sorted list of tubes as int values.
        """
        return sorted(list(set(
            [s.tube for case in self.cases for s in case.samples])))

    @property
    def group_count(self):
        """Get number of cases per group in case colleciton."""
        return collections.Counter(c.group for c in self.cases)

    @property
    def groups(self):
        return [c.group for c in self.cases]

    @property
    def labels(self):
        return [c.id for c in self.cases]

    @property
    def config(self):
        return {
            "selected_markers": self.selected_markers,
            "selected_tubes": self.selected_tubes,
            "filterconfig": self.filterconfig,
            "meta_path": self.meta_path,
            "data_path": self.data_path,
        }

    @property
    def counts(self):
        """Return a pandas dataframe with label group indexing a list of 1.
        This is used for combination with other datasets to filter for available data.
        """
        index = pd.MultiIndex.from_tuples(list(zip(self.labels, self.groups)), names=["label", "group"])
        return pd.DataFrame(1, index=index, columns=["count"])

    def add_filter_step(self, step: dict):
        self.filterconfig = [*self.filterconfig, step]

    def get_randnums(self, labels):
        """Get initial random numbers for single labels. Used when we want have
        multiple randomizations per case later on."""
        return {l: [0] for l in labels}

    def copy(self):
        data = [d.copy() for d in self.cases]
        return self.__class__(
            data,
            selected_tubes=self.selected_tubes,
            selected_markers=self.selected_markers)

    def set_markers(self):
        """Load markers from files."""
        for case in self:
            case.set_markers()

    def get_markers(self, tube):
        """Get a list of markers and their presence ratio in the current
        dataset."""
        marker_counts = collections.Counter(
            [m for c in self for m in c.get_tube_markers(tube)])
        ratios = pd.Series({m: k / len(self) for m, k in marker_counts.items()})
        return ratios

    def get_label(self, label):
        for case in self:
            if case.id == label:
                return case
        return None

    def label_to_group(self, label):
        """Return group of the given label."""
        case = self.get_label(label)
        return case.group if case else None

    def get_paths(self, label, randnum=0):
        case = self.get_label(label)
        tubes = self.selected_tubes or self.tubes
        paths = {
            t: str(case.get_tube(t).localpath) for t in tubes
        }
        return paths

    def map_groups(self, mapping):
        for case in self.cases:
            case.group = mapping.get(case.group, case.group)
        return self

    def sample(self, count: int, groups: List[str] = None) -> CaseCollection:
        """Select a sample from each group.
        Params:
            count: Number of cases in a single group.
            groups: Optionally limit to given groups.
        """
        group_labels = collections.defaultdict(list)
        for case in self.cases:
            group_labels[case.group].append(case.id)
        if groups is None:
            groups = group_labels.keys()
        labels = []
        for group in groups:
            glabels = group_labels[group]
            if len(glabels) > count:
                labels += random.sample(glabels, count)
            else:
                labels += glabels
        filtered, _ = self.filter_reasons(labels=labels)
        return filtered

    def filter_reasons(self, **kwargs) -> Tuple[CaseCollection, list]:
        """Filter dataset on given arguments. These are specified in
        case.filter_case.

        Returns:
            Filtered dataset and list of failed case ids and string reasons.
        """
        data = []
        failed = []
        for case in self:
            success, reasons = fc_case.filter_case(case, **kwargs)
            if success:
                ccase = case.copy()
                ccase.samples = fc_sample.filter_samples(ccase.samples, **kwargs)
                data.append(ccase)
            else:
                failed.append((case.id, reasons))
        filtered = self.__class__(data, **self.config)
        filtered.add_filter_step(kwargs)
        return filtered, failed

    def filter(self, **kwargs) -> CaseCollection:
        filtered, _ = self.filter_reasons(**kwargs)
        return filtered

    def create_split(self, num, stratify=True):
        """Split the data into two groups."""
        cases = pd.Series(self.cases)
        if stratify:
            trains = []
            tests = []
            for group, data in cases.groupby(by=lambda s: cases[s].group):
                if num < 1:
                    pivot = round(num * len(data))
                else:
                    pivot = int(num)
                data = data.tolist()
                random.shuffle(data)
                trains += data[:pivot]
                tests += data[pivot:]
        else:
            data = cases.reindex(np.random.permutation(cases.index))
            if num < 1:
                pivot = round(num * len(data))
            else:
                pivot = int(num)
            trains = data[:pivot].tolist()
            tests = data[pivot:].tolist()
        return (
            self.__class__(
                trains,
                selected_markers=self.selected_markers,
                selected_tubes=self.selected_tubes),
            self.__class__(
                tests,
                selected_markers=self.selected_markers,
                selected_tubes=self.selected_tubes),
        )

    def balance(self, num):
        """Balance classes to count given."""
        return self.balance_per_group({g: num for g in set(self.groups)})

    def balance_per_group(self, nums: dict) -> CaseCollection:
        """Randomly upsample groups based on numbers in dictionary. If a group
        is missing from dict, all cases in that group will be included."""
        case_groups = collections.defaultdict(list)
        for case in self.cases:
            case_groups[case.group].append(case)

        balanced = []
        for group_name, group_cases in case_groups.items():
            try:
                balanced += random.choices(group_cases, k=nums[group_name])
            except KeyError:
                balanced += group_cases

        return self.__class__(
            balanced,
            selected_markers=self.selected_markers,
            selected_tubes=self.selected_tubes
        )

    def shuffle(self):
        """Shuffle cases."""
        random.shuffle(self.cases)
        return self

    def __repr__(self):
        return f"<{self.__class__.__name__} {len(self)} cases>"

    def __iter__(self):
        return iter(self.cases)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, value):
        return self.cases[value]
