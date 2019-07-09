# Create datasets
from .dataset.case_dataset import CaseCollection
from .dataset.som_dataset import SOMDataset
from .som.base import load_som, save_som, SOM, SOMCollection
from . import utils, models, plots, som
from .mappings import (
    ALLOWED_MATERIALS, GROUP_MAPS, GROUPS, NAME_MAP,
    CHANNEL_CONFIGS,
    Material, Sureness
)
