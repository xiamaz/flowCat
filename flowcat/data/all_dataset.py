"""
Dataset combining information from FCS and other transformed sources.
"""
import re
import enum
import random
import functools
import logging
import collections

import pandas as pd

from . import case_dataset, loaders
from .. import utils


LOGGER = logging.getLogger(__name__)


class Datasets(enum.Enum):
    HISTO = enum.auto()
    SOM = enum.auto()
    FCS = enum.auto()

    @classmethod
    def from_str(cls, name):
        if isinstance(name, cls):
            return name
        name = name.upper()
        return cls[name]

    def get_class(self):
        """Get the associated class for the given value."""
        if self == self.HISTO:
            return HistoDataset
        elif self == self.SOM:
            return SOMDataset
        elif self == self.FCS:
            return case_dataset.CaseCollection
        else:
            raise TypeError(f"Type {self} has no associated class.")


def df_get_count(data, tubes):
    """Get count information from the given dataframe with labels as index.
    Args:
        data: dict of dataframes. Index will be used in the returned count dataframe.
        tubes: List of tubes as integers.
    Returns:
        Dataframe with labels as index and ratio of availability in the given tubes as value.
    """
    counts = None
    for tube in tubes:
        count = pd.DataFrame(
            1, index=data[tube].index, columns=["count"])
        count.reset_index(inplace=True)
        count.set_index(["label", "group"], inplace=True)
        count = count.loc[~count.index.duplicated(keep='first')]
        if counts is None:
            counts = count
        else:
            counts = counts.add(count, fill_value=0)
    counts = counts / len(tubes)
    return counts


class HistoDataset:
    """Information from histogram distribution experiments."""

    re_tube = re.compile(r".*[\/]tube(\d+).csv")

    def __init__(self, data, tubes):
        """Path to histogram dataset. Should contain a dataframe for
        each available tube."""
        self.counts = None
        self.data = data
        self.tubes = tubes
        self.set_counts(self.tubes)

    @classmethod
    def from_path(cls, path, tubes=None):
        data = cls.read_path(path)
        if tubes is None:
            tubes = list(data.keys())
        return cls(data, tubes)

    @classmethod
    def read_path(cls, path):
        """Read the given path and return a label mapped to either the actual
        data or a path."""
        tube_files = utils.URLPath(path).ls()
        data = {}
        for tfile in tube_files:
            match = cls.re_tube.match(str(tfile))
            if match:
                lpath = tfile.get()
                df = loaders.LoaderMixin.read_histo_df(lpath)
                data[int(match[1])] = pd.DataFrame(str(lpath), index=df.index, columns=["path"])
        return data

    def get_randnums(self, labels):
        """Get randnums for the given labels."""
        return {l: [0] for l in labels}

    def copy(self):
        data = {k: v.copy() for k, v in self.data.items()}
        return self.__class__(data, self.tubes.copy())

    def set_counts(self, tubes):
        self.counts = df_get_count(self.data, tubes)

    def get_path(self, tube, label, group):
        return self.data[tube].loc[label, group]

    def get_paths(self, label, randnum=0):
        return {k: v.loc[label, "path"].values[0] for k, v in self.data.items()}

    def __repr__(self):
        return f"<{self.__class__.__name__} {len(self.data)} tubes>"


class SOMDataset:
    """Infomation in self-organizing maps."""

    re_tube = re.compile(r".*[\/]\w+_t(\d+).csv")

    def __init__(self, data, tubes):
        """Path to SOM dataset. Should have another csv file with metainfo
        and individual SOM data inside the directory."""
        self.counts = None
        self.data = data
        self.tubes = tubes
        self.set_counts(self.tubes)

    @classmethod
    def from_path(cls, path, tubes=None):
        data = cls.read_path(path, tubes)
        if tubes is None:
            tubes = list(data.keys())
        return cls(data, tubes)

    @classmethod
    def read_path(cls, path, tubes):
        """Read the SOM dataset at the given path."""
        mappath = utils.URLPath(path)
        sompaths = utils.load_csv(mappath + ".csv")

        if tubes is None:
            tubes = cls.infer_tubes(mappath, sompaths.iloc[0, 0])

        soms = {}
        for tube in tubes:
            somtube = sompaths.copy()
            if "randnum" in somtube.columns:
                somtube["path"] = somtube.apply(
                    lambda r, t=tube: cls.get_path(mappath, r["label"], t, r["randnum"]), axis=1
                )
            else:
                somtube["path"] = somtube["label"].apply(
                    lambda l, t=tube: cls.get_path(mappath, l, t))
                somtube["randnum"] = 0

            somtube.set_index(["label", "randnum", "group"], inplace=True)
            soms[tube] = somtube

        return soms

    def get_paths(self, label, randnum=0):
        return {
            k: v.loc[[label, randnum], "path"].values[0]
            for k, v in self.data.items()
        }

    def get_randnums(self, labels):
        meta = next(iter(self.data.values()))
        return {l: meta.loc[l].index.get_level_values("randnum") for l in labels}

    @staticmethod
    def get_path(path, label, tube, random=None):
        if random is None:
            return str(path / f"{label}_t{tube}.csv")
        return str(path / f"{label}_{random}_t{tube}.csv")

    @classmethod
    def infer_tubes(cls, path, label):
        paths = path.glob(f"*{label}*.csv")
        tubes = sorted([int(m[1]) for m in [cls.re_tube.match(str(p)) for p in paths] if m])
        return tubes

    def copy(self):
        data = {k: v.copy() for k, v in self.data.items()}
        return self.__class__(data, self.tubes.copy())

    def set_counts(self, tubes):
        self.counts = df_get_count(self.data, tubes)


class CombinedDataset:
    """Combines information from different data sources."""

    def __init__(self, cases, datasets, mapping=None):
        """
        Args:
            fcspath: Path to fcs dataset. Necessary because of metainfo access.
            datasets: Additional datasets should be a dict between enums and types.
        """
        self.cases = cases
        self.datasets = datasets
        self.datasets[Datasets.FCS] = cases
        self.mapping = mapping

    @classmethod
    def from_paths(cls, casepath, paths):
        """Initialize from a list of paths with associated types."""
        cases = case_dataset.CaseCollection.from_dir(casepath)
        datasets = {
            Datasets.from_str(name): Datasets.from_str(name).get_class().from_path(path)
            for name, path in paths
        }
        return cls(cases, datasets)

    @property
    def fcs(self):
        return self.datasets[Datasets.FCS]

    @fcs.setter
    def fcs(self, value):
        self.datasets[Datasets.FCS] = value

    @property
    def label_groups(self):
        """Get all cases available in all datasets."""
        return list(zip(self.labels, self.groups))

    @property
    def labels(self):
        return self.fcs.labels

    @property
    def groups(self):
        if self.mapping is not None:
            return [self.mapping["map"].get(g, g) for g in self.fcs.groups]
        return self.fcs.groups

    def get_label_rand_group(self, dtypes):
        """Return list of label, randnum, group tuples
        using the dtypes requested."""
        lrandnums = [self.get_randnums(d) for d in dtypes]
        randnums = lrandnums[0]
        for randset in lrandnums[1:]:
            for label in randnums:
                randnums[label] = sorted(set(randset[label]) & set(randnums[label]))
        labels = [
            (label, randnum, group) for label, group in self.label_groups
            for randnum in randnums[label]
        ]
        return labels

    def get_randnums(self, dtype):
        return self.datasets[Datasets.from_str(dtype)].get_randnums(self.labels)

    def copy(self):
        datasets = {k: v.copy() for k, v in self.datasets.items()}
        return self.__class__(self.cases.copy(), datasets, mapping=self.mapping)

    def get(self, label, dtype, randnum=0):
        dtype = Datasets.from_str(dtype) if isinstance(dtype, str) else dtype
        return self.datasets[dtype].get_paths(label, randnum=randnum)

    def set_available(self, required):
        """Filter cases on availability in the required types."""
        required = [Datasets.from_str(r) if isinstance(r, str) else r for r in required]
        if self.fcs.selected_tubes:
            for req in required:
                self.datasets[req].set_counts(self.fcs.selected_tubes)
        sum_count = functools.reduce(
            lambda x, y: x.add(y, fill_value=0), (self.datasets[req].counts for req in required))
        sum_count = sum_count / len(required)
        labels = [l for l, _ in sum_count.loc[sum_count["count"] == 1, :].index.values.tolist()]
        self.fcs = self.fcs.filter(labels=labels)
        return self

    def set_mapping(self, mapping):
        self.mapping = mapping
        return self

    def filter(self, **kwargs):
        """Filter the dataset in place."""
        self.fcs = self.fcs.filter(**kwargs)
        return self

    def get_sample_weights(self, indices):
        labels = [l for l, *_ in indices]
        sample_weights = {}
        for case in self.fcs:
            if case.id in labels:
                sample_weights[case.id] = case.sureness
        return [sample_weights[i] for i, *_ in indices]


def split_dataset(data, train_num=None, test_labels=None, train_labels=None, seed=None):
    """Split data in stratified fashion by group.
    Args:
        data: Dataset to be split. Label should be contained in 'group' column.
        test_num: Ratio of samples in test per group or absolute number of samples in each group for test.
        seed: Seed for randomness in splitting.
    Returns:
        (train, test) with same columns as input data.
    """
    if seed:
        random.seed(seed)
    if test_labels is not None and train_labels is not None:
        LOGGER.info("Splitting based on provided label files")
        # simply load label lists from json files
        if not isinstance(test_labels, list):
            test_labels = utils.load_json(test_labels)
        if not isinstance(train_labels, list):
            train_labels = utils.load_json(train_labels)
    elif train_num is not None:
        group_labels = collections.defaultdict(list)
        for label, group in data.label_groups:
            group_labels[group].append(label)
        LOGGER.info("Stratifying in %d groups with a %f train split.", len(group_labels), train_num)
        # Splitting into test and train labels
        train_labels = []
        test_labels = []
        for group, labels in group_labels.items():
            random.shuffle(labels)
            if train_num < 1:
                k = int(len(labels) * train_num)
            else:
                if len(labels) < train_num:
                    raise ValueError(f"{group} size {len(labels)} < {train_num}")
                k = train_num
            train_labels += labels[:k]
            test_labels += labels[k:]
    else:
        raise RuntimeError(f"Specify either train_num or labels")

    LOGGER.info("Splitting into %d train and %d test cases. (seed %d)", len(train_labels), len(test_labels), seed)
    train = data.copy().filter(labels=train_labels)
    test = data.copy().filter(labels=test_labels)

    return train, test
