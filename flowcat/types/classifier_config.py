from copy import deepcopy
from dataclasses import dataclass, asdict, replace

import numpy as np
from flowcat import utils, io_functions
from flowcat.utils import classification_utils



def load_somclassifier_config(path: utils.URLPath) -> "SOMClassifierConfig":
    """Load somclassifier config from the given path."""
    return SOMClassifierConfig(**io_functions.load_json(path))


def save_somclassifier_config(config: "SOMClassifierConfig", path: utils.URLPath):
    """Save configuration to the given path."""
    io_functions.save_json(config.to_json(), path)


@dataclass
class SOMClassifierConfig:
    """Configuration information usable by SOM classifier."""

    tubes: dict  # Dict[TubeLabel, Dict[channels->List[Markers]|dims->Tuple of ints]]
    groups: list  # List of groups to be classified
    pad_width: int = 2
    mapping: dict = None
    cost_matrix: str = None
    train_epochs: int = 20
    train_batch_size: int = 32
    valid_batch_size: int = 128

    def to_json(self):
        return asdict(self)

    def copy(self):
        new_tubes = deepcopy(self.tubes)
        new_mapping = deepcopy(self.mapping)
        return replace(self, tubes=new_tubes, mapping=new_mapping)

    @property
    def output(self):
        return len(self.groups)

    @property
    def inputs(self):
        inputs = tuple(
            [*[dd + 2 * self.pad_width for dd in d["dims"][:-1]], len(d["channels"])] for d in self.tubes.values())
        return inputs

    def get_loss(self, modeldir=None):
        if self.cost_matrix is None:
            return "categorical_crossentropy"
        cost_matrix = np.load(modeldir / self.cost_matrix)
        return classification_utils.WeightedCategoricalCrossentropy(cost_matrix)
