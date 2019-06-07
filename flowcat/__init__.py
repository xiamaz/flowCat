# Create datasets
from .dataset.case_dataset import CaseCollection
from .som import create_som, load_som, save_som, SOM, SOMCollection
from . import utils, models, plots
from .mappings import (
    ALLOWED_MATERIALS, GROUP_MAPS, GROUPS, MARKER_THRESHOLD, NAME_MAP,
    CHANNEL_CONFIGS,
    Material, Sureness
)
