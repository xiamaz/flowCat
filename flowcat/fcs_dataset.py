from __future__ import annotations
from typing import List, Tuple

import numpy as np

from tensorflow import keras
from flowcat.dataset import fcs, case_dataset


class FCSSequence(keras.utils.Sequence):
    """Keras sequence backed by case collection with FCS samples."""

    def __init__(
        self,
        dataset: case_dataset.CaseCollection,
        tubes: List[str],
        binarizer,
        markers: dict,
        batch_size: int = 32,
    ):
        self.dataset = dataset
        self.binarizer = binarizer
        self.markers = markers
        self.tubes = tubes
        self.batch_size = batch_size
        self._cache = {}

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if idx in self._cache:
            return self._cache[idx]

        batch = self.dataset.data[idx * self.batch_size:(idx + 1) * self.batch_size]

        inputs = []
        for tube in self.tubes:
            markers = self.markers[tube]
            x_batch = np.array([
                s.get_tube(tube).get_data().align(
                    markers, missing_val=0.0, inplace=True).data.values
                for s in batch
            ])
            inputs.append(x_batch)

        y_labels = [s.group for s in batch]
        y_batch = self.binarizer.transform(y_labels)
        self._cache[idx] = inputs, y_batch
        return inputs, y_batch
