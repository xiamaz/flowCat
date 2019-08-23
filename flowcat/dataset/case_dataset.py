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
import pandas as pd
from sklearn import model_selection

from flowcat import mappings, utils

from . import case as fc_case
from . import sample as fc_sample


LOGGER = logging.getLogger(__name__)


class DatasetError(Exception):
    pass


def load_case_collection_from_caseinfo(data_path: utils.URLPath, meta_path: utils.URLPath) -> CaseCollection:
    """Load case collection from caseinfo json, as used in the MLL dataset."""
    metadata = utils.load_json(meta_path + ".json")
    try:
        metaconfig = utils.load_json(meta_path + "_config.json")
    except FileNotFoundError:
        metaconfig = {}
    data = [fc_case.caseinfo_to_case(d, data_path) for d in metadata]

    metaconfig["data_path"] = data_path
    metaconfig["meta_path"] = meta_path

    return CaseCollection(data, **metaconfig)


def save_case_collection_to_caseinfo(cases: CaseCollection, destination: utils.URLPath):
    """Save the metainformation to the given destination."""
    config_path = destination + "_config.json"
    utils.save_json(cases.config, config_path)
    data_path = destination + ".json"

    caseinfo = [fc_case.case_to_caseinfo(case, cases.data_path) for case in cases]
    utils.save_json(caseinfo, data_path)


@with_slots
@dataclass
class CaseCollection:
    """Iterable collection for cases. Base class."""

    cases: List[fc_case.Case]
    data_path: utils.URLPath
    meta_path: utils.URLPath

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
            data, selected_tubes=self.selected_tubes,
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
        # enforce same material
        material = case.get_possible_material(
            tubes, allowed_materials=mappings.ALLOWED_MATERIALS)
        paths = {
            t: str(case.get_tube(t, material=material).localpath) for t in tubes
        }
        return paths

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
        if stratify:
            labels = [d.group for d in self.cases]
        else:
            labels = None
        train, test = model_selection.train_test_split(
            self.cases, test_size=num, stratify=labels)
        return (
            self.__class__(
                train,
                selected_markers=self.selected_markers,
                selected_tubes=self.selected_tubes),
            self.__class__(
                test,
                selected_markers=self.selected_markers,
                selected_tubes=self.selected_tubes),
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} {len(self)} cases>"

    def __iter__(self):
        return iter(self.cases)

    def __len__(self):
        return len(self.cases)
