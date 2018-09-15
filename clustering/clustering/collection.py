import os
import random
import logging
from functools import reduce
import collections

import pandas as pd
from sklearn import preprocessing, model_selection

from .case import Case, CasePath, Material
from .utils import load_json, get_file_path, put_file_path


# Threshold for channel markers to be used in SOM
#
# It cases do not possess a marker required for the consensus SOM
# they will be ignored.
MARKER_THRESHOLD = 0.9

# materials allowed in processing
ALLOWED_MATERIALS = [Material.BONE_MARROW, Material.PERIPHERAL_BLOOD]

COLNAMES = ["label", "group", "infiltration"]

NO_INFILTRATION = ["normal"]

LOGGER = logging.getLogger(__name__)

INFONAME = "case_info.json"


class SelectedMarkers:

    def __init__(self, threshold=0.9):
        self._threshold = threshold
        self._marker_counts = None
        self._marker_ratios = None
        self._selected_markers = None

    @property
    def selected_markers(self):
        return self._selected_markers

    def fit(self, X: list, *_) -> list:
        # get a mapping between tubes and selected markers
        tube_markers = collections.defaultdict(list)
        for case in X:
            for tube, marker in case.tube_markers.items():
                tube_markers[tube] += marker

        # absolute marker counts
        self._marker_counts = {
            t: collections.Counter(m) for t, m in tube_markers.items()
        }

        # relative marker ratios across all cases
        self._marker_ratios = {
            t: {k: c[k] / len(X) for k in c}
            for t, c in self._marker_counts.items()
        }

        # select markers based on relative ratio above given threshold
        self._selected_markers = {
            t: [v for v, r in c.items() if r >= self._threshold]
            for t, c in self._marker_ratios.items()
        }
        return self

    def _predict(self, path: "CasePath") -> bool:
        """Check that all selected markers are contained in the given path
        information dict."""
        return path.has_markers(self._selected_markers[path.tube])

    def transform(self, X, *_):
        """Filter out all filepaths that do not contain required markers."""
        for single in X:
            single.filepaths = [
                path for path in single.filepaths if self._predict(path)
            ]
        return X

    def fit_transform(self, X, *_):
        return self.fit(X).transform(X)


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


class CaseIterable:
    """Iterable collection for cases. Base class."""

    def __init__(self):
        self._tubes = None
        self._groups = None
        self._current = None
        self._data = []


    @property
    def data(self) -> list:
        return self._data

    @data.setter
    def data(self, value: list):
        self._data = value
        # reset all cached computed objects
        self._tubes = None
        self._groups = None
        self._current = None

    @property
    def tubes(self):
        """Get list of unique tubes in dataset."""
        if self._tubes is None:
            self._tubes = list(set(
                [int(f.tube) for d in self.data for f in d.filepaths]
            ))
        return self._tubes

    @property
    def groups(self):
        """Get number of cases per group in case colleciton."""
        if self._groups is None:

            self._groups = collections.defaultdict(list)
            for data in self._data:
                self._groups[data.group].append(data)

        return self._groups

    @property
    def json(self):
        """Return dict represendation of content."""
        return [c.json for c in self]


def filter_materials(data: list) -> list:
    """Filter cases to remove not allowed materials.
    """
    for single_case in data:
        single_case.filepaths = [
            p for p in single_case.filepaths
            if p.material in ALLOWED_MATERIALS
        ]
    return data


class CaseCollection(IterableMixin, CaseIterable):
    """Get case information from info file and remove errors and provide
    overview information."""

    def __init__(self, inputpath: str, tubes=None, *args, **kwargs):
        """
        :param inputpath: Input directory containing cohorts and a info file.
        :param tubes: List of selected tubes.
        """
        super().__init__(*args, **kwargs)

        self._path = inputpath

        data = [
            Case(d, path=inputpath) for d in
            load_json(get_file_path(os.path.join(inputpath, INFONAME)))
        ]

        data = filter_materials(data)

        markers = SelectedMarkers()
        data = markers.fit_transform(data)
        self.markers = markers.selected_markers

        if tubes is not None:
            self.selected_tubes = tubes
        else:
            self.selected_tubes = self.tubes

        data = [
            d for d in data if d.has_tubes(self.selected_tubes)
        ]

        # ensure that data uses same material
        data = [
            d for d in data if d.same_material(self.selected_tubes)
        ]

        # check the dataset for duplicates
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

        self._data = deduplicated

    def create_view(
            self, labels=None, num=None, groups=None, infiltration=None, counts=None,
            **kwargs
    ):
        """Filter view to specified criteria and return a new view object."""
        # choose the basis for further filtering from either all cases
        # or a preselection of cases
        if groups:
            data = reduce(
                lambda x, y: x + y,
                [self.groups[g] for g in groups]
            )
        else:
            data = self._data

        if labels:
            data = [
                case for case in data if case.id in labels
            ]

        if infiltration:
            data = [
                d for d in data
                if d.infiltration >= infiltration or d.group in NO_INFILTRATION
            ]

        if counts:
            data = [
                d for d in data if d.has_count(counts, self.selected_tubes)
            ]

        # randomly sample num cases from each group
        if num:
            data = list(reduce(
                lambda x, y: x + y,
                [
                    random.sample(v, min(num, len(v)))
                    for v in [
                        [d for d in data if d.group == g]
                        for g in (groups or self.groups)
                    ]
                ]
            ))

        return CaseView(
            data, self.markers, tubes=self.selected_tubes, **kwargs
        )


class CaseView(IterableMixin, CaseIterable):
    """Filtered view into the base data. Perform all mutable
    actions on CaseView instead of CaseCollection."""

    def __init__(self, data, markers, tubes=None):
        super().__init__()
        self.data = data
        self.markers = markers
        self.selected_tubes = tubes

    def get_tube(self, tube: int) -> "TubeView":
        return TubeView(
            [
                d.get_tube(tube) for d in self.data
            ],
            self.markers[tube]
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
            CaseView(train, self.markers, tubes=self.selected_tubes),
            CaseView(test, self.markers, tubes=self.selected_tubes),
        )


class TubeView(IterableMixin):
    """List containing CasePath."""
    def __init__(self, data: [CasePath], markers: list):
        self.data = data
        self.markers = markers
        self._materials = None
        self._current = None


    @property
    def materials(self):
        if self._materials is None:
            self._materials = defaultdict(list)
            for casepath in self.data:
                self._materials[str(casepath.material)].append(casepath)

        return self._materials

    def export_results(self):
        """Export histogram results to pandas dataframe."""
        hists = [d.dict for d in self.data if d.result_success]
        failures = [d.fail_dict for d in self.data if not d.result_success]

        return (
            pd.DataFrame.from_records(hists),
            pd.DataFrame.from_records(failures)
        )
