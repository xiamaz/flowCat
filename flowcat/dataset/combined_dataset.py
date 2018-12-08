"""
Dataset combining information from FCS and other transformed sources.
"""
import random
import functools
import logging
import collections

from .dataset import Dataset
from .. import utils, loaders, mappings


LOGGER = logging.getLogger(__name__)


def get_path(dname, dpaths):
    """Get an existing dataset name."""
    for path in dpaths:
        dpath = utils.URLPath(path, dname)
        if dpath.exists():
            return dpath
    raise RuntimeError(f"Could not find {dname} in {dpaths}")


class CombinedDataset:
    """Combines information from different data sources."""

    def __init__(self, cases, datasets, group_names, mapping=None):
        """
        Args:
            fcspath: Path to fcs dataset. Necessary because of metainfo access.
            datasets: Additional datasets should be a dict between enums and types.
        """
        self.cases = cases
        self.datasets = datasets
        self.mapping = mapping
        self._group_names = group_names

    @classmethod
    def from_paths(cls, paths, tubes=None, **kwargs):
        """Initialize from a list of paths with associated types.
        Args:
            casepath: Path to fcs dataset also containing all metadata for cases.
            paths: Path to additional datasets with preprocessed data or other additional information.
            tubes: List of tubes to be used. This avoids searching for tubes inside the dataset.
        """
        datasets = {
            Dataset.from_str(name): Dataset.from_str(name).get_class().from_path(path, tubes=tubes)
            for name, path in paths.items()
        }
        assert Dataset.FCS in datasets, "Metadata needs to be provided by FCS dataset"
        cases = datasets[Dataset.FCS]
        return cls(cases, datasets, **kwargs)

    @classmethod
    def from_config(cls, pathconfig, config):
        """Initialize dataset from configuration dicts."""
        # define search paths from pathconfig
        datasets = pathconfig["input"]["paths"]

        # select the specific dataset from config
        dataset_names = config["dataset"]["names"]
        tubes = config["dataset"]["filters"]["tubes"]
        group_names = config["dataset"]["filters"]["groups"]

        paths = {
            dtype: get_path(name, datasets[dtype]) for dtype, name in dataset_names.items()
        }

        mapping = mappings.GROUP_MAPS[config["dataset"]["mapping"]]

        obj = cls.from_paths(paths, tubes=tubes, group_names=group_names, mapping=mapping)
        # filter using filters and only include cases available in all datasets
        obj.filter(**config["dataset"]["filters"])
        obj.set_available(dataset_names.keys())
        return obj

    @property
    def fcs(self):
        return self.datasets[Dataset.FCS]

    @fcs.setter
    def fcs(self, value):
        self.datasets[Dataset.FCS] = value

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

    @property
    def group_names(self):
        if self.mapping is not None:
            return self.mapping["groups"]
        else:
            return self._group_names

    def get_label_rand_group(self, dtypes):
        """Return list of label, randnum, group tuples using the dtypes requested."""
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
        return self.datasets[Dataset.from_str(dtype)].get_randnums(self.labels)

    def copy(self):
        datasets = {k: v.copy() for k, v in self.datasets.items()}
        return self.__class__(self.cases.copy(), datasets, mapping=self.mapping, group_names=self._group_names)

    def get(self, label, dtype, randnum=0):
        dtype = Dataset.from_str(dtype) if isinstance(dtype, str) else dtype
        return self.datasets[dtype].get_paths(label, randnum=randnum)

    def set_available(self, required):
        """Filter cases on availability in the required types."""
        required = [Dataset.from_str(r) for r in required]

        # count case availability in each tube
        if self.fcs.selected_tubes:
            for req in required:
                self.datasets[req].set_counts(self.fcs.selected_tubes)

        # sum counts across all datasts and calculate the ratio per case
        sum_count = functools.reduce(
            lambda x, y: x.add(y, fill_value=0), (self.datasets[req].counts for req in required))
        sum_count = sum_count / len(required)

        # get labels that are available in all datasets
        labels = [l for l, _ in sum_count.loc[sum_count["count"] == 1, :].index.values.tolist()]
        LOGGER.info("%d cases of %d samples total after availability filtering", len(labels), len(self.fcs))
        # filter metadata on the labels
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
        train_num: Ratio of samples in test per group or absolute number of samples in each group for test.
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
                k = int(len(labels) * train_num + 0.5)
            else:
                if len(labels) < train_num:
                    raise ValueError(f"{group} size {len(labels)} < {train_num}")
                k = train_num
            train_labels += labels[:k]
            test_labels += labels[k:]
    else:
        raise RuntimeError(f"Specify either train_num or labels")

    LOGGER.info(
        "Splitting into %d train and %d test cases. (seed %s)",
        len(train_labels), len(test_labels), str(seed))
    train = data.copy().filter(labels=train_labels)
    test = data.copy().filter(labels=test_labels)

    return train, test
