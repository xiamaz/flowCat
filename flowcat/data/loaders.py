"""
Data loaders for Sequence.
"""
import logging

import numpy as np
import pandas as pd
import fcsparser
from sklearn import preprocessing
from keras.utils import Sequence

from ..mappings import NAME_MAP
from ..caching import disk_cache


LOGGER = logging.getLogger(__name__)


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


class LoaderMixin:
    datacol = "sommappath"
    histocol = "histopath"
    fcscol = "fcspath"

    @staticmethod
    def load_data(path, tube):
        """Load the data associated with the given tube."""
        return pd.read_csv(str(path).format(tube=tube), index_col=0)


class CountLoader(LoaderMixin):

    """Load count information from 1d histogram analysis."""
    def __init__(self, tube, width, version="mapcount", data=None):
        self.tube = tube
        self.version = version
        self.data = data
        self.width = width

    @staticmethod
    def read_dataframe(path, tube):
        dfdata = pd.read_csv(path.format(tube=tube), index_col=0)
        dfdata["group"] = dfdata["group"].apply(lambda g: NAME_MAP.get(g, g))
        dfdata.set_index(["label", "group"], inplace=True)
        non_number_cols = [c for c in dfdata.columns if not c.isdigit()]
        dfdata.drop(non_number_cols, inplace=True, axis=1)
        return dfdata

    @classmethod
    def create_inferred(cls, data, tube, version="mapcount"):
        dfdata = None
        if version == "dataframe":
            dfdata = cls.read_dataframe(data[cls.histocol].iloc[0], tube=tube)
            width = dfdata.shape[1]
        elif version == "mapcount":
            mapdata = cls.load_data(data[cls.datacol].iloc[0], tube)
            width = mapdata.shape[0]
        return cls(tube=tube, width=width, version=version, data=dfdata)

    @property
    def shape(self):
        return (self.width, )

    def __call__(self, data):
        if self.version == "mapcount":
            count_list = []
            for path in data[self.datacol]:
                mapdata = self.load_data(path, self.tube)
                countdata = select_drop_counts(
                    mapdata, sel_count="count_prev")["count_prev"]
                count_list.append(countdata.values)
            counts = np.stack(count_list)
        elif self.version == "dataframe":
            if self.data is None:
                self.data = self.read_dataframe(
                    data[self.datacol].iloc[0], self.tube)
            label_group = [x for x in zip(data.index, data["orig_group"])]
            sel_rows = self.data.loc[label_group, :]
            missing = sel_rows.loc[sel_rows["1"].isna(), :]
            if not missing.empty:
                LOGGER.error(missing)
                raise RuntimeError()
            counts = sel_rows.values
        return counts


class FCSLoader(LoaderMixin):
    """Directly load FCS data associated with the given ids."""
    def __init__(self, tubes, channels=None, subsample=200):
        self.tubes = tubes
        self.channels = channels
        self.subsample = subsample

    @classmethod
    def create_inferred(cls, data, tubes, subsample=200, channels=None, *args, **kwargs):
        testdata = cls._load_data(
            data[cls.fcscol].iloc[0], subsample, tubes=tubes, channels=channels
        )
        channels = list(testdata.columns)
        return cls(tubes=tubes, channels=channels, subsample=subsample, *args, **kwargs)

    @property
    def shape(self):
        return (self.subsample * len(self.tubes), len(self.channels))

    @staticmethod
    @disk_cache
    def _load_tube_data(path):
        _, data = fcsparser.parse(path, data_set=0, encoding="latin-1")

        data.drop([c for c in data.columns if "nix" in c], axis=1, inplace=True)

        data = data / 1024.0

        # data = pd.DataFrame(scaler.transform(data), columns=data.columns)

        # data = pd.DataFrame(
        #     preprocessing.MinMaxScaler().fit_transform(
        #         preprocessing.StandardScaler().fit_transform(data)),
        #     columns=data.columns)

        cols = [c + s for c in data.columns for s in ["", "sig"]]
        sig_cols = [c for c in cols if c.endswith("sig")]
        data = pd.concat(
            [data, pd.DataFrame(1, columns=sig_cols, index=data.index)], axis=1)
        data = data.loc[:, cols]

        return data

    @classmethod
    def _load_data(cls, pathdict, subsample, tubes, channels=None):
        datas = []
        for tube in tubes:
            data = cls._load_tube_data(pathdict[tube])
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
            mapped_fcs.append(self._load_data(path, self.subsample, self.tubes, self.channels))
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
        refdata = cls.load_data(data[cls.datacol].iloc[0], tube)
        nodes, channels = refdata.shape
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

    @staticmethod
    @disk_cache
    def _load_sommap(path, tube, sel_count, pad_width, gridsize):
        mapdata = Map2DLoader.load_data(path, tube)
        mapdata = select_drop_counts(mapdata, sel_count)
        mapdata = reshape_dataframe(mapdata, m=gridsize, n=gridsize, pad_width=pad_width)
        return mapdata

    def _get_mapdata(self, pathlist):
        map_list = []
        for path in pathlist:
            data = self._load_sommap(path, self.tube, self.sel_count, self.pad_width, self.gridsize)
            map_list.append(data)
        return np.stack(map_list)

    def __call__(self, data):
        """Output specified format."""
        return self._get_mapdata(list(data[self.datacol]))


class SOMMapDataset(LoaderMixin, Sequence):
    """Dataset for creating and yielding batches of data for keras model.

    Data can be generated by random draw or alternatively in sequence.
    """

    def __init__(
            self, data, xoutputs,
            batch_size=32, draw_method="shuffle", epoch_size=None,
            groups=None, group_nums=None, sample_weights=False,
    ):
        """
        Args:
            data: DataFrame containing labels and paths to data.
            xoutputs: List of output generator objects taking batches of the filepath dataframe.
            batch_size: Number of cases in a single batch.
            draw_method: Method to select cases in a single batch.
                valid: [
                    'shuffle', # shuffle all data and return batches
                    'sequential',  # return data in sequence
                    'balanced'  # present balanced representation of data in one epoch
                    'groupnums'  # specified number of samples per group
                ]
            epoch_size: Number of samples in a single epoch. Is data length if None or 0.
            groups: List of groups to transform the labels into binary matrix.
            group_nums: Number of samples per group for balanced sampling. If
                not given, will evenly distribute the epoch size among all groups.
            sample_weights: Use sample weights in training.
        Returns:
            SOMMapDataset object.
        """
        self.batch_size = batch_size
        self.draw_method = draw_method
        self.sample_weights = sample_weights

        self._all_data = data
        if groups is None:
            groups = list(self._all_data["group"].unique())
        self.groups = groups
        self.group_nums = group_nums

        self._data = self._sample_data(data, epoch_size)
        self.epoch_size = self._data.shape[0]

        self._xoutputs = xoutputs

    def _sample_data(self, data, epoch_size=None):
        if self.draw_method == "shuffle":
            selection = data.sample(frac=1)
        elif self.draw_method == "sequential":
            selection = data
        elif self.draw_method == "balanced":
            sample_num = int((epoch_size or len(data)) / len(self.groups))
            selection = data.groupby("group").apply(lambda x: x.sample(
                n=sample_num, replace=True)).reset_index(0, drop=True).sample(frac=1)
        elif self.draw_method == "groupnum":
            selection = data.groupby("orig_group").apply(lambda x: x.sample(
                n=self.group_nums[x.name], replace=True)).reset_index(0, drop=True).sample(frac=1)
        else:
            raise RuntimeError(
                f"Unknown draw method: {self.draw_method}. "
                "Valid options are: ['shuffle', 'sequential', 'balanced']")
        return selection

    @property
    def xshape(self):
        """Return shape of xvalues. Should be a list of shapes describing each input.
        """
        return [x.shape for x in self._xoutputs]

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

        xdata = [x(batch_data) for x in self._xoutputs]

        ydata = batch_data["group"]
        ybinary = preprocessing.label_binarize(ydata, classes=self.groups)
        if self.sample_weights:
            sample_weights = batch_data["sureness"].values
            return xdata, ybinary, sample_weights
        return xdata, ybinary

    def on_epoch_end(self):
        """Randomly reshuffle data after end of epoch."""
        self._data = self._sample_data(self._all_data, self.epoch_size)
