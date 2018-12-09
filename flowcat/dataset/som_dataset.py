import re

from .. import utils


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
        self.counts = utils.df_get_count(self.data, tubes)
