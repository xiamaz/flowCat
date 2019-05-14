import re
import logging

from .. import utils, configuration
from .. import som


LOGGER = logging.getLogger(__name__)


class SOMDataset:
    """Infomation in self-organizing maps."""

    re_tube = re.compile(r".*[\/]\w+_t(\d+).csv")

    def __init__(self, data, tubes, path=None):
        """Path to SOM dataset. Should have another csv file with metainfo
        and individual SOM data inside the directory."""
        self.counts = None
        self.data = data
        self.tubes = tubes
        self.set_counts(self.tubes)

        self.path = path

    @classmethod
    def from_path(cls, path, tubes=None):
        data = cls.read_path(path, tubes)
        if tubes is None:
            tubes = list(data.keys())
        return cls(data, tubes, path=path)

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

            somtube.set_index(["label", "randnum"], inplace=True)
            soms[tube] = somtube

        return soms

    def get_paths(self, label, randnum=0):
        return {
            k: v.loc[[label, randnum], "path"].values[0]
            for k, v in self.data.items()
        }

    def get_config(self):
        """Read configuration, which is to be located in dataset_name/config.toml"""
        if self.path is not None:
            configpath = utils.URLPath(self.path) / "config.toml"
            if configpath.exists():
                return configuration.SOMConfig.from_file(configpath)

        return None

    def save_config(self, path, pathconfig=None):
        config = self.get_config()
        if config is None:
            LOGGER.warning("No configuration file found for SOM")
            return
        config.to_file(path / "config.toml")

        # save SOM reference too
        reference = config("reference")
        if reference:
            if pathconfig is not None:
                reference = utils.get_path(reference, [pathconfig("output", "som-reference")])
            refdata = som.load_som(reference, self.tubes)
            som.save_som_dict(refdata, path / "reference", suffix=True)

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
        return self.__class__(data, self.tubes.copy(), path=self.path)

    def set_counts(self, tubes):
        self.counts = utils.df_get_count(self.data, tubes)
