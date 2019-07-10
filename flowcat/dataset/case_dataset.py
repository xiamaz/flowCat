"""
Classes for managing collections of case and tubecase objects.
"""
from __future__ import annotations
import random
import logging
import collections
import datetime
from typing import Union, List

import pandas as pd
from sklearn import model_selection

from .case import Case, filter_case, filter_tubesamples
from .. import mappings, utils
from ..utils import load_json, load_csv, URLPath


LOGGER = logging.getLogger(__name__)


def get_meta(path, how):
    """Choose strategy on how to select for metadata."""
    if how == "latest":
        case_info = sorted(path.glob("*.json"))[-1]
    elif how == "oldest":
        case_info = sorted(path.glob("*.json"))[0]
    else:
        # use how as the complete name
        case_info = path / how
        assert case_info.exists()
    return case_info


def load_meta(path, transfun=None):
    if str(path).endswith(".json"):
        data = load_json(path)
    elif str(path).endswith(".csv"):
        data = load_csv(path, index_col=None)
    else:
        raise TypeError(f"Unsupported filetype for metadata {path}")
    if transfun is not None:
        data = transfun(data)
    return data


class NoTubeSelectedError(Exception):
    pass


class IterableMixin:
    """Implement iterable stuff but needs to have the data attribute."""
    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current >= len(self):
            raise StopIteration
        else:
            self._current += 1
            return self[self._current - 1]


class CaseIterable(IterableMixin):
    """Iterable collection for cases. Base class."""

    def __init__(
            self,
            data: Union[CaseIterable, List[Case]],
            selected_markers=None,
            selected_tubes=None,
            filterconfig: Union[tuple, list] = ()
    ):
        """
        Args:
            data: Case iterable or a list of cases
            selected_markers: Dictionary of tubes to list of marker channels.
            selected_tubes: List of tube numbers.
            filterconfig: List of dicts containing previous kwargs used in filtering Cases
        """
        self._current = None
        self._data = []
        self.filterconfig = filterconfig

        if isinstance(data, CaseIterable):
            self.data = data.data
            # use metainformation if they are given
            if selected_markers is None:
                selected_markers = data.selected_markers
            if selected_tubes is None:
                selected_tubes = data.selected_tubes
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError(f"Cannot initialize from {type(data)}")

        self.selected_markers = selected_markers
        self.selected_tubes = selected_tubes

    @property
    def data(self) -> list:
        return self._data

    @data.setter
    def data(self, value: list):
        self._data = value
        # reset all cached computed objects
        self._current = None

    @property
    def tubes(self):
        """Get list of unique tubes in dataset.
        Returns:
            A sorted list of tubes as int values.
        """
        return sorted(list(set(
            [int(f.tube) for d in self.data for f in d.filepaths])))

    @property
    def group_count(self):
        """Get number of cases per group in case colleciton."""
        return collections.Counter(c.group for c in self)

    @property
    def groups(self):
        return [c.group for c in self]

    @property
    def labels(self):
        return [c.id for c in self]

    @property
    def json(self):
        """Return dict represendation of content."""
        return [c.json for c in self]

    @property
    def config(self):
        return {
            "selected_markers": self.selected_markers,
            "selected_tubes": self.selected_tubes,
            "filterconfig": self.filterconfig,
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
        data = [d.copy() for d in self.data]
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

    def sample(self, count: int, groups: List[str] = None):
        """Select a sample from each group.
        Params:
            count: Number of cases in a single group.
            groups: Optionally limit to given groups.
        """
        group_labels = collections.defaultdict(list)
        for case in self.data:
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

    def filter_reasons(self, **kwargs):
        data = []
        failed = []
        for case in self:
            success, reasons = filter_case(case, **kwargs)
            if success:
                ccase = case.copy()
                ccase.filepaths = filter_tubesamples(ccase.filepaths, **kwargs)
                data.append(ccase)
            else:
                failed.append((case.id, reasons))
        filtered = self.__class__(data, **self.config)
        filtered.add_filter_step(kwargs)
        return filtered, failed

    def __repr__(self):
        return f"<{self.__class__.__name__} {len(self)} cases>"


class CaseCollection(CaseIterable):
    """Get case information from info file and remove errors and provide
    overview information."""

    def __init__(self, data, path="", metapath="", *args, **kwargs):
        try:
            super().__init__(data, *args, **kwargs)
        except ValueError:
            raise ValueError(f"Paths to directories should be instantiated with {self.__class__.__name__}.from_path")
        self.path = utils.URLPath(path)
        self.metapath = utils.URLPath(metapath)

    @classmethod
    def from_path(cls, inputpath, how="latest", metapath=None, transfun=None, **kwargs):
        """
        Initialize on datadir with info json.
        Args:
            inputpath: Input directory containing cohorts and a info file.
        """
        inputpath = URLPath(inputpath)
        if metapath is None:
            metapath = get_meta(inputpath, how=how)
        else:
            metapath = URLPath(metapath)
        metadata = load_meta(metapath, transfun=transfun)
        data = [Case(d, path=inputpath) for d in metadata]

        return cls(data, inputpath, metapath, **kwargs)

    @property
    def config(self):
        return {
            **super().config,
            "path": str(self.path),
            "metapath": str(self.metapath)
        }

    def get_tube(self, tube: int) -> "TubeView":
        if self.selected_markers is None:
            raise NoTubeSelectedError
        return TubeView(
            [
                d.get_tube(tube) for d in self
            ],
            markers=self.selected_markers[tube],
            tube=tube,
        )

    def download_all(self):
        """Download selected tubes for cases into the download folder
        location."""
        if self.selected_markers is None:
            raise NoTubeSelectedError
        for case in self:
            for tube in self.selected_tubes:
                # touch the data to load it
                print("Loaded df with shape: ", case.get_tube(tube).data.shape)

    def create_split(self, num, stratify=True):
        """Split the data into two groups."""
        if stratify:
            labels = [d.group for d in self.data]
        else:
            labels = None
        train, test = model_selection.train_test_split(
            self.data, test_size=num, stratify=labels)
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


class TubeView(IterableMixin):
    """List containing CasePath."""
    def __init__(self, data, markers, tube):
        assert None not in data, "None contained in passed list"
        self.markers = markers
        self.tube = tube
        self._data = data
        self._materials = None
        self._current = None
        self._labels = None
        self._groups = None

    @property
    def data(self):
        return self._data

    @property
    def materials(self):
        if self._materials is None:
            self._materials = [d.material for d in self]

        return self._materials

    @property
    def labels(self):
        if self._labels is None:
            self._labels = [d.parent.id for d in self]
        return self._labels

    @property
    def groups(self):
        if self._groups is None:
            self._groups = [d.parent.group for d in self]
        return self._groups

    @property
    def marker_ratios(self):
        """Get a list of markers and their presence ratio in the current
        dataset."""
        marker_counts = collections.Counter(
            [m for c in self.data for m in c.markers])
        ratios = pd.Series({m: k / len(self) for m, k in marker_counts.items()})
        return ratios

    def export_results(self):
        """Export histogram results to pandas dataframe."""
        hists = [d.dict for d in self.data if d.result_success]
        failures = [d.fail_dict for d in self.data if not d.result_success]

        return (
            pd.DataFrame.from_records(hists),
            pd.DataFrame.from_records(failures)
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} {len(self)} cases>"
