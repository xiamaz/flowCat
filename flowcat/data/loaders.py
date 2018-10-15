"""
Data loaders for Sequence.
"""
import pathlib
import hashlib
import functools
import pickle
import logging

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


def args_hasher(*args, **kwargs):
    """Use at own discretion. Will simply concatenate all input args as
    strings to generate keys."""
    hasher = hashlib.blake2b()
    hashstr = "".join(str(a) for a in args) + "".join(str(k) + str(v) for k, v in kwargs.items())
    hasher.update(hashstr.encode())
    return hasher.hexdigest()


def disk_cache(fun):
    """Cache function results depending on arguments on disk in a cache directory."""

    cachepath = pathlib.Path(CACHEDIR) / fun.__name__
    cachepath.mkdir(parents=True, exist_ok=True)

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        hashed = args_hasher(*args, **kwargs)
        filepath = cachepath / hashed
        if filepath.exists():
            with open(str(filepath), "rb") as f:
                result = pickle.load(f)
        else:
            result = fun(*args, **kwargs)
            with open(str(filepath), "wb") as f:
                pickle.dump(result, f)
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


def loader_builder(constructor, *args, **kwargs):
    """Partial function to delay data loading in Loader building."""
    def build(data, **nargs):
        return constructor(data=data, *args, **{**kwargs, **nargs})  # later unpacking overrides earlier
    return build


class LoaderMixin:
    somcol = "sommappath"
    histocol = "histopath"
    fcscol = "fcspath"

    @staticmethod
    @disk_cache
    def read_som(path, tube, sel_count=None, gridsize=None, pad_width=None):
        """Load the data associated with the given tube."""
        mapdata = utils.load_csv(str(path).format(tube=tube))
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


class CountLoader(LoaderMixin):

    """Load count information from 1d histogram analysis."""
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
        dfdata = cls.load_histos(data.iloc[[0], :], tube=tube, datatype=datatype, datacol=datacol)
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
            # get label group tuples usable for indexing the count dataframe
            dfpath = data[cls.histocol].iloc[0].format(tube)
            label_group = [x for x in zip(data.index, data["orig_group"])]
            # get the selected counts
            counts = cls.read_histo_df(dfpath)[label_group, :].values
        elif datatype == "mapcount":
            assert datacol is not None, "mapcount needs datacol"
            count_list = []
            for path in data[cls.somcol]:
                mapdata = cls.read_histo_som(path, tube, datacol)
                count_list.append(mapdata.values)
            counts = np.stack(count_list)
        else:
            raise TypeError(f"Unknown type {datatype}")
        return counts

    def __call__(self, data):
        return self.load_histos(data, self.tube, self.datatype, self.datacol)


class FCSLoader(LoaderMixin):
    """Directly load FCS data associated with the given ids."""
    def __init__(self, tubes, channels=None, subsample=200):
        self.tubes = tubes
        self.channels = channels
        self.subsample = subsample

    @classmethod
    def create_inferred(cls, data, tubes, subsample=200, channels=None, *args, **kwargs):
        testdata = cls.load_fcs(
            data[cls.fcscol].iloc[0], subsample, tubes=tubes, channels=channels
        )
        channels = list(testdata.columns)
        return cls(tubes=tubes, channels=channels, subsample=subsample, *args, **kwargs)

    @property
    def shape(self):
        return (self.subsample * len(self.tubes), len(self.channels))

    @classmethod
    def load_fcs(cls, pathdict, subsample, tubes, channels=None):
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

    def __call__(self, data):
        mapped_fcs = []
        for path in data[self.fcscol]:
            mapped_fcs.append(self.load_fcs(path, self.subsample, self.tubes, self.channels))
        return np.stack(mapped_fcs)


class Map2DLoader(LoaderMixin):
    """2-Dimensional SOM maps for 2D-Convolutional processing."""
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
        refdata = cls.read_som(data[cls.somcol].iloc[0], tube)
        nodes, _ = refdata.shape
        gridsize = int(np.ceil(np.sqrt(nodes)))
        non_count_channels = [c for c in refdata.columns if "count" not in c]
        return {"gridsize": gridsize, "channels": non_count_channels}

    @property
    def shape(self):
        return (
            self.gridsize + self.pad_width * 2,
            self.gridsize + self.pad_width * 2,
            len(self.channels) + bool(self.sel_count)
        )

    def load_som(self, path):
        return self.read_som(path, self.tube, self.sel_count, self.gridsize, self.pad_width)

    def __call__(self, data):
        """Output specified format."""
        pathlist = list(data[self.somcol])
        map_list = []
        for path in pathlist:
            data = self.load_som(path)
            map_list.append(data)
        mapdata = np.stack(map_list)
        return mapdata


class DatasetSequence(LoaderMixin, Sequence):
    """Dataset usable as sequence input for keras.

    Data can be generated by random draw or alternatively in sequence.
    """

    def __init__(
            self, data, output_spec,
            batch_size=32, draw_method="shuffle", epoch_size=None,
            groups=None, number_per_group=None, sample_weights=False,
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
        self._all_data = data
        self._output_spec = [x(data) for x in output_spec]

        self.batch_size = batch_size
        self.draw_method = draw_method
        self.sample_weights = sample_weights
        self.number_per_group = number_per_group

        if groups is None:
            groups = list(self._all_data["group"].unique())
        self.groups = groups

        self._data = self._sample_data(data, epoch_size)
        self.epoch_size = self._data.shape[0]

    def _sample_data(self, data, epoch_size=None, number_per_group=None):
        """
        Get version of the dataset to be used for one epoch. This can be whole
        dataset if we just want to process everything sequentially.

        This is a view of our dataset to be used for one epoch. Batches are
        always generated by index number in the keras sequence way to enable
        parallel processing.

        Args:
            data: Dataframe with the data to be worked on.
            epoch_size: Number of sample in one epoch, only needed for some draw_methods. Only used by 'balanced'.
            number_per_group: Number of samples to be used per group. Only used by 'groupnum'.
        Returns:
            Slice of the data to be selected.
        """
        # changing nothing will always return the same slices of data in each
        # epoch given the same batch numbers
        if self.draw_method == "sequential":
            selection = data
        # randomize the batches in subsequent epochs by reshuffling in each new
        # epoch
        elif self.draw_method == "shuffle":
            selection = data.sample(frac=1)
        # always choose a similar number from all cohorts, smaller cohorts can
        # be duplicated to match the numbers in the larger cohorts
        # if epoch_size is provided the number per group is split evenly
        elif self.draw_method == "balanced":
            sample_num = int((epoch_size or len(data)) / len(self.groups))
            selection = data.groupby("group").apply(lambda x: x.sample(
                n=sample_num, replace=True)).reset_index(0, drop=True).sample(frac=1)
        # directly provide number needed per group
        elif self.draw_method == "groupnum":
            assert number_per_group is not None, "groupnum needs number_per_group"
            selection = data.groupby("orig_group").apply(lambda x: x.sample(
                n=number_per_group[x.name], replace=True)).reset_index(0, drop=True).sample(frac=1)
        else:
            raise RuntimeError(
                f"Unknown draw method: {self.draw_method}. "
                "Valid options are: ['shuffle', 'sequential', 'balanced', 'groupnum']")
        return selection

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
        return self._data.index.values.tolist()

    @property
    def ylabels(self):
        return self._data["group"]

    def get_batch_by_label(self, label):
        '''Return batch by label'''
        return self[self.labels.index(label)]

    def __len__(self):
        """Return the number of batches generated."""
        return int(np.ceil(self.epoch_size / float(self.batch_size)))

    def __getitem__(self, idx):
        """Get a single batch by id."""
        batch_data = self._data.iloc[idx * self.batch_size: (idx + 1) * self.batch_size, :]

        # load data using output specs
        xdata = [x(batch_data) for x in self._output_spec]

        ydata = batch_data["group"]
        ybinary = preprocessing.label_binarize(ydata, classes=self.groups)
        if self.sample_weights:
            sample_weights = batch_data["sureness"].values
            return xdata, ybinary, sample_weights
        return xdata, ybinary

    def on_epoch_end(self):
        """Resample data after end of epoch."""
        self._data = self._sample_data(self._all_data, self.epoch_size, self.number_per_group)
