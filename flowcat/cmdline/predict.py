from flowcat import utils, som_dataset, classifier, mappings, io_functions
from flowcat.classifier import predictions as fc_predictions


def predict(data: utils.URLPath, model: utils.URLPath, output: utils.URLPath, labels: utils.URLPath = None):
    """Generate predictions and plots for a single case.

    Args:
        data: SOM dataset.
        model: Path to model containing CNN and SOMs.
        output: Destination for plotting.
        labels: List of case ids to be filtered for generating predictions.
    """
    dataset = som_dataset.SOMDataset.from_path(data)
    if labels:
        labels = io_functions.load_json(labels)
        dataset = dataset.filter(labels=labels)

    model = classifier.SOMClassifier.load(model)
    data_sequence = model.create_sequence(dataset, 128)

    _, labels = model.predict(data_sequence)

    true_labels = data_sequence.true_labels
    for map_name, mapping in [("unmapped", {"groups": model.config.groups, "map": {}}), *mappings.GROUP_MAPS.items()]:
        print(f"--- MAPPING: {map_name} ---")
        if len(mapping["groups"]) > len(model.config.groups):
            continue
        fc_predictions.generate_all_metrics(true_labels, labels, mapping, output / map_name)
