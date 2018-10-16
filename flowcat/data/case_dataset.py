"""
Classes for managing collections of case and tubecase objects.
"""
import os
import random
import logging
from functools import reduce
import collections

import pandas as pd
from sklearn import model_selection

from .case import Case, TubeSample, Material
from ..utils import load_json, URLPath


# Threshold for channel markers to be used in SOM
#
# It cases do not possess a marker required for the consensus SOM
# they will be ignored.
MARKER_THRESHOLD = 0.9
EMPTY_TAG = "nix"  # Part of channel name if no antibodies have been loaded

# materials allowed in processing
ALLOWED_MATERIALS = [Material.BONE_MARROW, Material.PERIPHERAL_BLOOD]

COLNAMES = ["label", "group", "infiltration"]

NO_INFILTRATION = ["normal"]

LOGGER = logging.getLogger(__name__)

INFONAME = "case_info.json"


def deduplicate_cases_by_sureness(data):
    """Remove duplicates by taking the one with the higher sureness score."""
    label_dict = collections.defaultdict(list)
    for single in data:
        label_dict[single.id].append(single)
    deduplicated = []
    for cases in label_dict.values():
        cases.sort(key=lambda c: c.sureness, reverse=True)
        if len(cases) == 1 or cases[0].sureness > cases[1].sureness:
            deduplicated.append(cases[0])
        else:
            LOGGER.warning(
                "DUP both removed: %s (%s), %s (%s)\nSureness: %s %s",
                cases[0].id, cases[0].group, cases[1].id, cases[1].group,
                cases[0].sureness_description, cases[1].sureness_description,
            )
    return deduplicated


class IterableMixin(object):
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

    def __init__(self, data, selected_markers=None, selected_tubes=None):
        """
        Args:
            data: List of cases.
            selected_markers: Dictionary of tubes to list of marker channels.
            selected_tubes: List of tube numbers.
        """
        self._current = None
        self._data = []

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
            raise TypeError(f"Cannot initialize from {type(data)}")

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
    def counts(self):
        """Return a pandas dataframe with label group indexing a list of 1.
        This is used for combination with other datasets to filter for available data.
        """
        index = pd.MultiIndex.from_tuples(list(zip(self.labels, self.groups)), names=["label", "group"])
        return pd.DataFrame(1, index=index, columns=["count"])

    def copy(self):
        data = [d.copy() for d in self.data]
        return self.__class__(
            data, selected_tubes=self.selected_tubes,
            selected_markers=self.selected_markers)

    def set_counts(self, tubes):
        tcopy = tubes.copy()
        # tubes have been set
        if self.selected_tubes is None:
            raise AssertionError
        # test for list equality
        for tube in self.selected_tubes:
            try:
                tcopy.remove(tube)
            except ValueError:
                raise AssertionError
        if tcopy:
            raise AssertionError

    def get_markers(self, tube):
        """Get a list of markers and their presence ratio in the current
        dataset."""
        marker_counts = collections.Counter(
            [m for c in self for m in c.get_tube(tube).markers])
        ratios = pd.Series({m: k / len(self) for m, k in marker_counts.items()})
        return ratios

    def get_label(self, label):
        for case in self:
            if case.id == label:
                return case
        return None

    def get_paths(self, label):
        case = self.get_label(label)
        tubes = self.selected_tubes or self.tubes
        # enforce same material
        material = case.get_possible_material(
            tubes, allowed_materials=ALLOWED_MATERIALS)
        paths = {
            t: str(case.get_tube(t, material=material).localpath) for t in tubes
        }
        return paths

    def filter(
            self,
            tubes=None,
            labels=None,
            num=None,
            groups=None,
            infiltration=None,
            counts=None,
            materials=None,
            selected_markers=None,
    ):
        """Get filtered version of the data."""

        # set defaults
        if materials is None:
            materials = ALLOWED_MATERIALS
        if tubes is None:
            tubes = self.tubes

        # choose the basis for further filtering from either all cases
        # or a preselection of cases
        data = self._data

        if groups:
            data = [case for case in data if case.group in groups]

        if labels:
            data = [case for case in data if case.id in labels]

        if infiltration:
            data = [d for d in data if d.infiltration >= infiltration or d.group in NO_INFILTRATION]

        # create copy since we will start mutating the objects
        data = [d.copy() for d in data]

        ndata = []
        # filter cases by allowed materials and counts in fcs files
        for case in data:
            ccase = case.copy()
            ccase.used_material = ccase.get_possible_material(tubes, materials)

            has_count = all(ccase.get_tube(t, min_count=counts) for t in tubes)
            if ccase.used_material and has_count:
                # remove filepaths with fewer than count
                if counts:
                    ccase.filepaths = [fp for fp in ccase.filepaths if fp.count >= counts]
                ndata.append(ccase)
        data = ndata

        if selected_markers is None:
            tubemarkers = {
                t: collections.Counter(m for c in data for m in c.get_tube(t).markers)
                for t in tubes
            }
            selected_markers = {
                t: [m for m, n in v.items() if n / len(data) > MARKER_THRESHOLD and "nix" not in m]
                for t, v in tubemarkers.items()
            }

        # randomly sample num cases from each group
        if num:
            data = [
                d for glist in [
                    random.sample(v, min(num, len(v)))
                    for v in [
                        [d for d in data if d.group == g]
                        for g in (groups or set(self.groups))
                    ]
                ] for d in glist
            ]
        return self.__class__(data, selected_markers=selected_markers, selected_tubes=tubes)

    def __repr__(self):
        return f"<{self.__class__.__name__} {len(self)} cases>"


class CaseCollection(CaseIterable):
    """Get case information from info file and remove errors and provide
    overview information."""

    def __init__(self, data, path="", *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.path = path

    @classmethod
    def from_dir(cls, inputpath):
        """
        Initialize on datadir with info json.
        Args:
            inputpath: Input directory containing cohorts and a info file.
        """
        metapath = URLPath(inputpath, INFONAME)
        data = [
            Case(d, path=inputpath) for d in load_json(metapath.get())
        ]

        # check the dataset for duplicates
        data = deduplicate_cases_by_sureness(data)

        return cls(data, inputpath)

    def filter(self, *args, **kwargs):
        """Adding more metadata to the filtered result."""
        result = super().filter(*args, **kwargs)
        view = CaseView(result)
        return view


class CaseView(CaseIterable):
    """Filtered view into the base data. Perform all mutable
    actions on CaseView instead of CaseCollection."""

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        assert self.selected_markers is not None
        assert self.selected_tubes is not None

    def get_tube(self, tube: int) -> "TubeView":
        return TubeView(
            [
                d.get_tube(tube) for d in self
            ],
            self.selected_markers[tube]
        )

    def download_all(self):
        """Download selected tubes for cases into the download folder
        location."""
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
            CaseView(
                train,
                selected_markers=self.selected_markers,
                selected_tubes=self.selected_tubes),
            CaseView(
                test,
                selected_markers=self.selected_markers,
                selected_tubes=self.selected_tubes),
        )


class TubeView(IterableMixin):
    """List containing CasePath."""
    def __init__(self, data, markers):
        self.markers = markers
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
