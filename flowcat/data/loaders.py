"""
Data loaders for Sequence.
"""
import pathlib
import hashlib
import functools
import pickle
import logging
import random
import collections

import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import Sequence

from ..mappings import NAME_MAP, GROUP_MAPS
from . import case as ccase
from . import case_dataset as cc
from .. import utils


LOGGER = logging.getLogger(__name__)
CACHEDIR = "cache"


def stringify(*args, **kwargs):
    """Transform args and kwargs into a single string, which can be used for hashing"""
    return "".join(str(a) for a in args) + "".join(str(k) + str(v) for k, v in kwargs.items())


def args_hasher(*args, **kwargs):
    """Use at own discretion. Will simply concatenate all input args as
    strings to generate keys."""
    hasher = hashlib.blake2b()
    hashstr = stringify(*args, **kwargs)
    hasher.update(hashstr.encode())
    return hasher.hexdigest()


def disk_cache(fun):
    """Cache function results depending on arguments on disk in a cache directory."""

    cachepath = pathlib.Path(CACHEDIR) / fun.__name__
    cachepath.mkdir(parents=True, exist_ok=True)

    # all hashes that have been saved this time can be skipped later
    prev_hashed = []

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        hashed = args_hasher(*args, **kwargs)
        filepath = cachepath / hashed
        if hashed in prev_hashed:
            with open(str(filepath), "rb") as f:
                result = pickle.load(f)
        elif filepath.exists():
            with open(str(filepath), "rb") as f:
                result = pickle.load(f)
            prev_hashed.append(hashed)
        else:
            result = fun(*args, **kwargs)
            with open(str(filepath), "wb") as f:
                pickle.dump(result, f)
            prev_hashed.append(hashed)
        return result

    return wrapper


def mem_cache(fun):
    """Cache function output inside the calling object."""

    cache = {}

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        hashed = args_hasher(*args, **kwargs)
        if hashed in cache:
            result = cache[hashed]
        else:
            # LOGGER.debug("Cache miss: %s size: %d", fun.__name__, len(cache))
            result = fun(*args, **kwargs)
            cache[hashed] = result
        return result

    return wrapper


def normalize_data(data):
    data["data"] = data["data"].apply(
        lambda t: [
            pd.DataFrame(
                preprocessing.MinMaxScaler().fit_transform(d),
                columns=d.columns
            ) for d in t]
    )
    return data


def reshape_dataframe(data, m=10, n=10, pad_width=0):
    """Reshape dataframe 2d matrix with channels as additional dimension.
    Optionally pad the data.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.reshape(data, (m, n, -1))
    if pad_width:
        data = np.pad(data, pad_width=[
            (pad_width, pad_width),
            (pad_width, pad_width),
            (0, 0),
        ], mode="wrap")
    return data


def select_drop_counts(data, sel_count=None):
    """Select and preprocess count channel. If sel_count is None, drop
    all count channels.
    """
    countnames = ["counts", "count_prev"]
    if sel_count is not None:
        data[sel_count] = np.sqrt(data[sel_count])
        # rescale 0-1
        data[sel_count] = data[sel_count] / max(data[sel_count])

    data.drop(
        [c for c in countnames if c != sel_count], axis=1, inplace=True,
        errors="ignore"
    )
    return data


def label_binarize(data, classes):
    transformed = preprocessing.label_binarize(data, classes=classes)
    if len(classes) == 2:
        transformed = np.hstack((1 - transformed, transformed))
    elif len(classes) < 2:
        raise RuntimeError("Output less than 2 classes.")
    return transformed


def loader_builder(constructor, *args, **kwargs):
    """Partial function to delay data loading in Loader building."""
    def build(data, **nargs):
        return constructor(data=data, *args, **{**kwargs, **nargs})  # later unpacking overrides earlier
    return build


class LoaderMixin:
    @staticmethod
    def read_som(path, tube, sel_count=None, gridsize=None, pad_width=None):
        """Load the data associated with the given tube."""
        mapdata = utils.load_csv(utils.URLPath(str(path).format(tube=tube)))
        mapdata = select_drop_counts(mapdata, sel_count)
        if gridsize is not None:
            mapdata = reshape_dataframe(mapdata, m=gridsize, n=gridsize, pad_width=pad_width)
        return mapdata

    @staticmethod
    @mem_cache
    def read_histo_df(path):
        """Read count data histogram."""
        dfdata = utils.load_csv(path)
        # translate legacy naming scheme with mixed german names to
        # standardized english names
        dfdata["group"] = dfdata["group"].apply(lambda g: NAME_MAP.get(g, g))
        # set index on label and group
        dfdata.set_index(["label", "group"], inplace=True)
        # remove columns not denoting count numbers
        non_number_cols = [c for c in dfdata.columns if not c.isdigit()]
        dfdata.drop(non_number_cols, inplace=True, axis=1)

        return dfdata

    @staticmethod
    @mem_cache
    def read_histo_som(path, tube, count_col):
        """Get the count column from a specific SOM map."""
        assert count_col is not None, "Count col has to be specified."
        return LoaderMixin.read_som(path, tube, sel_count=count_col)[count_col]

    @staticmethod
    @disk_cache
    def read_fcs(path, transform="rescale"):
        """
        Read a FCS file from the given path and applyt the given transformation.
        Args:
            path: path to FCS file.
            transform: Apply specific transformation to the input data.
        """
        data = ccase.FCSData(path).drop_empty().data
        if transform == "none":
            data = data
        elif transform == "rescale":
            data = data / 1024.0
        elif transform == "zscores_scaled":
            data = ccase.FCSMinMaxScaler().fit_transform(
                ccase.FCSStandardScaler().fit_transform(data))
        elif transform == "zscores":
            data = ccase.FCSStandardScaler().fit_transform(data)
        else:
            raise TypeError(f"Unknown transform type {transform}")

        cols = [c + s for c in data.columns for s in ["", "sig"]]
        sig_cols = [c for c in cols if c.endswith("sig")]
        data = pd.concat(
            [data, pd.DataFrame(1, columns=sig_cols, index=data.index)], axis=1)
        data = data.loc[:, cols]
        return data

    def load_data(self, path):
        raise RuntimeError("Implement me.")

    def __call__(self, label_groups, dataset):
        """Output specified format."""
        batchdata = []
        for label, randnum, _ in label_groups:
            pathdict = dataset.get(label, self.dtype, randnum=randnum)
            data = self.load_data(pathdict)  # pylint: disable=assignment-from-no-return
            batchdata.append(data)
        batch = np.stack(batchdata)
        return batch


class CountLoader(LoaderMixin):
    """Load count information from 1d histogram analysis."""

    dtype = "HISTO"

    def __init__(self, tube, width, datatype="mapcount", datacol=None):
        """
        Args:
            tube: Tube associated with the input data.
            width: Number of bins in the histogram.
            datatype: Source of the data.
                'dataframe' - Histogram dataframe as generated by the old pipeline.
                'mapcount' - Use the count field from SOM maps.
            datacol: Needed for mapcount to specify the column containing the counts.
        """
        assert datatype != "mapcount" or datacol is not None, "mapcount type needs specified data colomn."

        self.tube = tube
        self.datatype = datatype
        self.width = width
        self.datacol = datacol

    @classmethod
    def create_inferred(cls, data, tube, datatype="mapcount", datacol=None):
        label_group = data.label_groups[0]
        pathdict = [(label_group, data.get(label_group[0], cls.dtype)[tube])]
        dfdata = cls.load_histos(pathdict, tube=tube, datatype=datatype, datacol=datacol)
        width = dfdata.shape[1]
        return cls(tube=tube, width=width, datatype=datatype)

    @property
    def shape(self):
        return (self.width, )

    @classmethod
    def load_histos(cls, data, tube, datatype, datacol):
        """Load list of paths into a np array containing counts.
        Args:
            data: Selected rows from paths dataframe. Should always be a dataframe, not a list.
        """
        if datatype == "dataframe":
            path = data[0][1]  # first element in tuple of (label, group), path
            # get the selected counts
            indices = [i for i, _, in data]
            counts = cls.read_histo_df(path).loc[indices, :].values
        elif datatype == "mapcount":
            assert datacol is not None, "mapcount needs datacol"
            count_list = []
            for _, path in data:
                mapdata = cls.read_histo_som(path, tube, datacol)
                count_list.append(mapdata.values)
            counts = np.stack(count_list)
        else:
            raise TypeError(f"Unknown type {datatype}")
        return counts

    def __call__(self, label_groups, dataset):
        """Output specified format."""
        label_group_paths = [
            ((l, g), dataset.get(l, self.dtype, randnum=r)[self.tube]) for l, r, g in label_groups
        ]
        batch = self.load_histos(
            label_group_paths, self.tube, self.datatype, self.datacol)
        return batch


class FCSLoader(LoaderMixin):
    """Directly load FCS data associated with the given ids."""

    dtype = "FCS"

    def __init__(self, tubes, channels=None, subsample=200, randomize=False,):
        self.tubes = tubes
        self.channels = channels
        self.subsample = subsample
        self.randomize = randomize

    @classmethod
    def create_inferred(cls, data, tubes, subsample=200, channels=None, *args, **kwargs):
        testdata = cls.load_fcs(
            data.get(data.labels[0], "FCS"), subsample, tubes=tubes, channels=channels
        )
        channels = list(testdata.columns)
        return cls(tubes=tubes, channels=channels, subsample=subsample, *args, **kwargs)

    @property
    def shape(self):
        return (self.subsample * len(self.tubes), len(self.channels))

    @classmethod
    def load_fcs_randomized(cls, pathdict, subsample, tubes, channels=None):
        datas = []
        for tube in tubes:
            data = cls.read_fcs(pathdict[tube])
            data = data.sample(n=subsample)
            datas.append(data)

        merged = pd.concat(datas, sort=False)
        merged = merged.fillna(0)
        if channels:
            return merged[channels].values
        else:
            return merged

    @classmethod
    @mem_cache
    @disk_cache
    def load_fcs(cls, *args, **kwargs):
        return cls.load_fcs_randomized(*args, **kwargs)

    def load_data(self, path):
        if self.randomize:
            return self.load_fcs_randomized(path, self.subsample, self.tubes, self.channels)
        else:
            return self.load_fcs(path, self.subsample, self.tubes, self.channels)


class Map2DLoader(LoaderMixin):
    """2-Dimensional SOM maps for 2D-Convolutional processing."""

    dtype = "SOM"

    def __init__(self, tube, gridsize, channels, sel_count=None, pad_width=0, cached=False):
        """Object to transform input rows into the specified format."""
        self.tube = tube
        self.gridsize = gridsize
        self.channels = channels
        self.sel_count = sel_count
        self.pad_width = pad_width
        self._cache = {}

    @classmethod
    def create_inferred(cls, data, tube, *args, **kwargs):
        """Create with inferred information."""
        return cls(tube=tube, *args, **cls.infer_size(data, tube), **kwargs)

    @classmethod
    def infer_size(cls, data, tube=1):
        """Infer size of input from data."""
        label = data.labels[0]
        path = data.get(label, cls.dtype)[tube]
        # read the SOM file
        refdata = cls.read_som(path, tube)
        # number of nodes in rows
        nodes, _ = refdata.shape
        # always a m x m grid
        gridsize = int(np.ceil(np.sqrt(nodes)))
        # get all non count channel names
        non_count_channels = [c for c in refdata.columns if "count" not in c]
        return {"gridsize": gridsize, "channels": non_count_channels}

    @property
    def shape(self):
        return (
            self.gridsize + self.pad_width * 2,
            self.gridsize + self.pad_width * 2,
            len(self.channels) + bool(self.sel_count)
        )

    def load_data(self, pathdict):
        return self.read_som(pathdict[self.tube], self.tube, self.sel_count, self.gridsize, self.pad_width)


class DatasetSequence(LoaderMixin, Sequence):
    """Dataset usable as sequence input for keras.

    Data can be generated by random draw or alternatively in sequence.
    """

    def __init__(
            self, data, output_spec,
            batch_size=32, draw_method="shuffle", epoch_size=None,
            number_per_group=None, sample_weights=False,
    ):
        """
        Args:
            data: DataFrame containing labels and paths to data.
            output_spec: List of output generator objects taking batches of the filepath dataframe.
            batch_size: Number of cases in a single batch.
            draw_method: Method to select cases in a single batch.
                valid: [
                    'shuffle', # shuffle all data and return batches
                    'sequential',  # return data in sequence
                    'balanced'  # present balanced representation of data in one epoch
                    'groupnums'  # specified number of samples per group
                ]
            epoch_size: Number of samples in a single epoch. If not specified will be the length of the input data.
            groups: List of groups to transform the labels into binary matrix.
            number_per_group: Number of samples per group for balanced sampling. If
                not given, will evenly distribute the epoch size among all groups.
            sample_weights: Use sample weights in training.
        Returns:
            SOMMapDataset object.
        """
        self._data = data
        self._output_spec = [x(data) for x in output_spec]

        self.batch_size = batch_size
        self.draw_method = draw_method

        self.sample_weights = sample_weights
        self.avg_sample_weight = np.mean(
            self._data.get_sample_weights(self._data.label_groups))

        self.groups = self._data.group_names

        self.epoch_size, self.label_groups, self.number_per_group = self._sample_data(
            data, epoch_size, number_per_group)

    @property
    def output_dtypes(self):
        return [o.dtype for o in self._output_spec]

    def _sample_data(self, data, epoch_size=None, number_per_group=None):
        """
        Get version of the dataset to be used for one epoch. This can be whole
        dataset if we just want to process everything sequentially.

        Args:
            data: Dataset containing path information. Should contain all
                possible values.
            epoch_size: Number of samples in a single epoch
            number_per_group: Number of samples per group
        Returns:
            Number of labels, Selection of labels to be used, Number of labels per group (can be None)
        """
        # get a list of labels, random_labels, groups
        selection = data.get_label_rand_group(self.output_dtypes)

        if self.draw_method == "sequential":
            LOGGER.debug("Sequentially sampling data. Will cache batches in memory.")
        elif self.draw_method == "shuffle":
            LOGGER.debug("Shuffling all data.")
            selection = random.sample(selection, k=len(selection))
        elif self.draw_method == "balanced":
            LOGGER.debug("Randomly drawing data with replacement.")
            label_groups = collections.defaultdict(list)
            for label, randnum, group in selection:
                label_groups[group].append((label, randnum, group))

            if not number_per_group:
                LOGGER.debug("No number per group provided. Splitting epoch size evenly between all groups.")
                sample_num = int((epoch_size or len(selection)) / len(label_groups))
                number_per_group = {g: sample_num for g in label_groups}
            else:
                assert all(g in number_per_group for g in label_groups), \
                    "Not all groups contained in provided numbers per group"

            selection = []
            for group, labels in label_groups.items():
                sample_num = number_per_group[group]
                LOGGER.debug("Taking %d from %s.", sample_num, group)
                selection += random.choices(labels, k=sample_num)
        else:
            raise RuntimeError(
                f"Unknown draw method: {self.draw_method}. "
                "Valid options are: ['shuffle', 'sequential', 'balanced', 'groupnum']")
        return len(selection), selection, number_per_group

    @property
    def xshape(self):
        """Return shape of xvalues. Should be a list of shapes describing each input.
        """
        return [x.shape for x in self._output_spec]

    @property
    def yshape(self):
        """Return shape of yvalues. If we only have 2 classes use simple
        binary encoding.
        """
        return len(self.groups)

    @property
    def shape(self):
        """Return tuple of xshape and yshape."""
        return self.xshape, self.yshape

    @property
    def labels(self):
        return [l for l, _, _ in self.label_groups]

    @property
    def ylabels(self):
        return [g for _, _, g in self.label_groups]

    @property
    def randnums(self):
        return [r for _, r, _ in self.label_groups]

    def get_batch_by_label(self, label):
        '''Return batch by label'''
        return self[self.labels.index(label)]

    def __len__(self):
        """Return the number of batches generated."""
        return int(np.ceil(self.epoch_size / float(self.batch_size)))

    @mem_cache
    def __getitem__(self, idx):
        """Get a single batch by id."""
        batch_data = self.label_groups[idx * self.batch_size: (idx + 1) * self.batch_size]

        # load data using output specs
        xdata = [x(batch_data, self._data) for x in self._output_spec]

        ydata = [g for _, _, g in batch_data]
        ybinary = label_binarize(ydata, classes=self.groups)

        if self.sample_weights:
            sample_weights = np.array(self._data.get_sample_weights(batch_data)) / self.avg_sample_weight * 5
            return xdata, ybinary, sample_weights
        return xdata, ybinary
