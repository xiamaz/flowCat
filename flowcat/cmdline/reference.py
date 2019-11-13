import json
import logging
from flowcat import utils, sommodels, io_functions


LOGGER = logging.getLogger(__name__)


def setup_logging():
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger("flowcat", handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def reference(
        data: utils.URLPath,
        meta: utils.URLPath,
        output: utils.URLPath,
        labels: utils.URLPath,
        tensorboard: bool = False,
        trainargs: json.loads = None,
        selected_markers: json.loads = None):
    """Train new reference SOM from random using data filtered by labels."""
    setup_logging()

    dataset = io_functions.load_case_collection(data, meta)
    labels = io_functions.load_json(labels)
    dataset = dataset.filter(labels=labels)

    if trainargs is None:
        trainargs = {
            "marker_name_only": False,
            "max_epochs": 10,
            "batch_size": 50000,
            "initial_radius": 16,
            "end_radius": 2,
            "radius_cooling": "linear",
            # "marker_images": sommodels.fcssom.MARKER_IMAGES_NAME_ONLY,
            "map_type": "toroid",
            "dims": (32, 32, -1),
            "scaler": "MinMaxScaler",
        }

    if selected_markers is None:
        selected_markers = dataset.selected_markers

    tensorboard_dir = None
    if tensorboard:
        tensorboard_dir = output / "tensorboard"

    print("Creating SOM model with following parameters:")
    print(trainargs)
    print(selected_markers)
    model = sommodels.casesom.CaseSom(
        tubes=selected_markers,
        tensorboard_dir=tensorboard_dir,
        modelargs=trainargs,
    )
    print(f"Training SOM")
    model.train(dataset)

    print(f"Saving trained SOM to {output}")
    io_functions.save_casesom(model, output)
