"""
Abstractions for flow cytometry data. This will wrap both the metadata dict and
the pandas dataframe returned by fcsparser.
"""
import pandas as pd
import fcsparser


class FCSData(object):
    __slots__ = (
        "_meta", "data", "ranges"
    )

    def __init__(self, meta, data, ranges=None):
        """Create a new FCS object.

        Args:
            meta: Dict containing FCS header information.
            data: Pandas dataframe with channels in columns.
            tube: Number of tube of the data.
        Returns:
            FCSData object.
        """

        self._meta = meta
        self.data = data

        if ranges is None:
            self.ranges = self._get_ranges_from_pnr(self._meta)
        else:
            self.ranges = ranges

    @classmethod
    def from_path(cls, path):
        """Load fcs data from the given filepath."""
        meta, data = fcsparser.parse(path)
        return cls(meta, data)

    @property
    def meta(self):
        return self._meta

    def copy(self):
        return self.__class__(self.meta.copy(), self.data.copy(), ranges=self.ranges.copy())

    def _get_ranges_from_pnr(self, metadata):
        """Get ranges from metainformation."""
        pnr = {
            c: {
                "min": 0,
                "max": int(metadata[f"$P{i + 1}R"])
            } for i, c in enumerate(self.data.columns)
        }
        pnr = pd.DataFrame.from_dict(pnr, orient="columns", dtype="float32")
        return pnr

    def __repr__(self):
        """Print string representation of the input file."""
        nevents, nchannels = self.data.shape
        return f"<FCS :: {nevents} events :: {nchannels} channels>"
