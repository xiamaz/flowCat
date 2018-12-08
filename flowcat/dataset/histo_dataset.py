import re
import logging
import pandas as pd


from .. import loaders, utils


LOGGER = logging.getLogger(__name__)


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
        self.counts = utils.df_get_count(self.data, tubes)

    def get_path(self, tube, label, group):
        return self.data[tube].loc[label, group]

    def get_paths(self, label, randnum=0):
        return {k: v.loc[label, "path"].values[0] for k, v in self.data.items()}

    def __repr__(self):
        return f"<{self.__class__.__name__} {len(self.data)} tubes>"
