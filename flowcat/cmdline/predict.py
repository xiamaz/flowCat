from flowcat import utils, classifier, io_functions
from flowcat.constants import GROUP_MAPS
from flowcat.classifier import predictions as fc_predictions, som_dataset


def predict(
        data: utils.URLPath,
        model: utils.URLPath,
        output: utils.URLPath,
        labels: utils.URLPath = None,
        metrics: bool = True,
):
    """Generate predictions and plots for a single case.

    Args:
        data: SOM dataset.
        model: Path to model containing CNN and SOMs.
        output: Destination for plotting.
        labels: List of case ids to be filtered for generating predictions.
    """
    print(f"Loaded cases from {data}")
    dataset = som_dataset.SOMDataset.from_path(data)
    if labels:
        labels = io_functions.load_json(labels)
        dataset = dataset.filter(labels=labels)

    model = classifier.SOMClassifier.load(model)
    data_sequence = model.create_sequence(dataset, 128)

    values, pred_labels = model.predict_generator(data_sequence)

    pred_json = {
        id: dict(zip(model.config.groups, value.tolist())) for id, value in zip(dataset.labels, values)
    }

    io_functions.save_json(pred_json, output / "prediction.json")

    if metrics:
        true_labels = data_sequence.true_labels
        map_config = [("unmapped", {"groups": model.config.groups, "map": {}}), *GROUP_MAPS.items()]
        for map_name, mapping in map_config:
            print(f"--- MAPPING: {map_name} ---")
            if len(mapping["groups"]) > len(model.config.groups):
                continue
            fc_predictions.generate_all_metrics(true_labels, pred_labels, mapping, output / map_name)
