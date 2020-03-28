import json
import logging
from flowcat import utils, sommodels, io_functions
from flowcat.constants import DEFAULT_REFERENCE_SOM_ARGS


LOGGER = logging.getLogger(__name__)


def setup_logging():
    handlers = [
        utils.create_handler(logging.StreamHandler(), level=logging.INFO)
    ]

    utils.add_logger("flowcat", handlers, level=logging.DEBUG)
    utils.add_logger(LOGGER, handlers, level=logging.DEBUG)


def reference(
        data: utils.URLPath,
        output: utils.URLPath,
        labels: utils.URLPath = None,
        meta: utils.URLPath = None,
        tensorboard: bool = False,
        trainargs: json.loads = None,
        selected_markers: json.loads = None):
    """Train new reference SOM from random using data filtered by labels."""
    setup_logging()

    dataset = io_functions.load_case_collection(data, meta)
    if labels:
        labels = io_functions.load_json(labels)
        print(f"Filtering dataset using {len(labels)} labels")
        dataset = dataset.filter(labels=labels)
        if len(dataset) != len(labels):
            raise RuntimeError("Filtered number of samples does not match number of labels.")

    if trainargs is None:
        trainargs = DEFAULT_REFERENCE_SOM_ARGS

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
