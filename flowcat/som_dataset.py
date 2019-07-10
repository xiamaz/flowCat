from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from keras.utils import Sequence

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

    @property
    def labels(self):
        return np.array([s.group for s in self.data])

    @property
    def group_counts(self):
        return {
            group: len(data)
            for group, data in self.data.groupby(by=lambda s: self.data[s].group)
        }

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
            train = pd.concat(trains)
            validate = pd.concat(valids)
        else:
            data = self.data.reindex(np.random.permutation(data.index))
            pivot = round(ratio * len(data))
            train = data[pivot:]
            validate = data[:pivot]

        train.reset_index(drop=True, inplace=True)
        train = train.reindex(np.random.permutation(train.index))
        validate.reset_index(drop=True, inplace=True)
        validate = validate.reindex(np.random.permutation(validate.index))

        return (
            self.__class__(train, tubes=self.tubes, dims=self.dims, channels=self.channels),
            self.__class__(validate, tubes=self.tubes, dims=self.dims, channels=self.channels),
        )

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"<SOMDataset {len(self)} cases>"


class SOMSequence(Sequence):

    def __init__(self, dataset: SOMDataset, binarizer, batch_size: int = 32, tube=1):
        self.dataset = dataset
        self.tube = tube
        self.batch_size = batch_size
        self.binarizer = binarizer

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        batch = self.dataset.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = np.array([s.get_tube(self.tube).np_array() for s in batch])
        y_labels = [s.group for s in batch]
        y_batch = self.binarizer.transform(y_labels)
        return x_batch, y_batch
