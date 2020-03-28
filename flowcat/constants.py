from .mappings import GROUPS
from .classifier import SOMClassifierConfig

DEFAULT_REFERENCE_SOM_ARGS = {
    "marker_name_only": False,
    "max_epochs": 10,
    "batch_size": 50000,
    "initial_radius": 16,
    "end_radius": 2,
    "radius_cooling": "linear",
    # "marker_images": sommodels.fcssom.MARKER_IMAGES_NAME_ONLY,
    "map_type": "toroid",
    "dims": [32, 32, -1],
    "scaler": "MinMaxScaler",
}

DEFAULT_TRANSFORM_SOM_ARGS = {
    "max_epochs": 4,
    "batch_size": 50000,
    "initial_radius": 4,
    "end_radius": 1,
}

DEFAULT_CLASSIFIER_CONFIG = SOMClassifierConfig(
    tubes=None,
    groups=GROUPS,
    pad_width=2,
    mapping=None,
    cost_matrix=None,
    train_epochs=20,
    train_batch_size=32,
    valid_batch_size=128,
)

DEFAULT_CLASSIFIER_ARGS = {
    "split_ratio": 0.9,
    "split_stratify": True,
    "balance": {
        "CLL": 4000,
        "MBL": 2000,
        "MCL": 1000,
        "PL": 1000,
        "LPL": 1000,
        "MZL": 1000,
        "FL": 1000,
        "HCL": 1000,
        "normal": 6000,
    },
    "config": DEFAULT_CLASSIFIER_CONFIG,
}

DEFAULT_TRAIN_ARGS = {
    "reference": DEFAULT_REFERENCE_SOM_ARGS,
    "transform": DEFAULT_TRANSFORM_SOM_ARGS,
    "classifier": DEFAULT_CLASSIFIER_ARGS,
}
