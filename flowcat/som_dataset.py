from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from flowcat import utils
from flowcat.som.base import load_som, SOMCollection, SOM


@dataclass
class SOMCase:
    """Wrapper for SOM data, associating it with additional classification
    metadata."""

    label: str
    group: str
    som: SOMCollection

    def get_tube(self, tube: int) -> SOM:
        return self.som.get_tube(tube)

    def __repr__(self):
        return f"<SOMCase {self.label} {self.group}"


def load_som_cases(row, path, tubes):
    sompath = path / row["label"]
    som = load_som(sompath, subdirectory=False, tube=tubes)
    return SOMCase(som=som, group=row["group"], label=row["label"])


class SOMDataset:
    """Simple wrapper for reading dataset metadata."""

    def __init__(self, data, tubes, dims, channels):
        self.data = data
        self.tubes = tubes
        self.dims = dims
        self.channels = channels

    @classmethod
    def from_path(cls, path):
        path = utils.URLPath(path)
        config = utils.load_json(path + ".json")
        metadata = utils.load_csv(path + ".csv")
        som_cases = metadata.apply(load_som_cases, axis=1, args=(path, config["tubes"]))
        return cls(data=som_cases, **config)

    def get_tube(self, tube: int) -> List[SOM]:
        return [s.get_tube(tube) for s in self.data]

    def split(self, ratio: float, stratified: bool = True) -> Tuple[SOMDataset, SOMDataset]:
        if stratified:
            trains = []
            valids = []
            for group, data in self.data.groupby(by=lambda s: self.data[s].group):
                data.reset_index(drop=True, inplace=True)
                data = data.reindex(np.random.permutation(data.index))
                pivot = round(ratio * len(data))
                trains.append(data[:pivot])
                valids.append(data[pivot:])
                print(group, data)
            train = pd.concat(trains)
            validate = pd.concat(valids)
        else:
            data = self.data.reindex(np.random.permutation(data.index))
            pivot = round(ratio * len(data))
            train = data[pivot:]
            validate = data[:pivot]

        train.reset_index(drop=True, inplace=True)
        validate.reset_index(drop=True, inplace=True)

        return train, validate

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"<SOMDataset {len(self)} cases>"
