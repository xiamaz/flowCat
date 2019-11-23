from flowcat import utils, classifier, som_dataset, mappings, io_functions


def check_dataset_groups(dataset, groups):
    """Check that all groups given are actually contained in the dataset.
    """
    dataset_groups = {d.group for d in dataset}
    return set(groups) == dataset_groups


def load_datasets(
        data: utils.URLPath,
        split_ratio=0.9,
        mapping=None,
        groups=None,
        balance=None):
    """Prepare dataset splitting and optional upsampling.

    Args:
        split_ratio: Ratio of training set to test set.
        mapping: Optionally map existing groups to new groups contained in mapping.
        groups: List of groups to be used.
        balance: Dict or value to upsample the training dataset.
    """
    dataset = som_dataset.SOMDataset.from_path(data)

    if mapping:
        dataset = dataset.map_groups(mapping)

    if groups:
        dataset = dataset.filter(groups=groups)
        if not check_dataset_groups(dataset, groups):
            raise RuntimeError(f"Group mismatch: Not all groups in {groups} are in dataset.")

    train, validate = dataset.create_split(split_ratio, stratify=True)

    if balance:
        train = train.balance_per_group(balance)
    return train, validate


def train(data: utils.URLPath, output: utils.URLPath):
    """Train a new classifier using SOM data."""
    groups = mappings.GROUPS
    tubes = ("1", "2", "3")
    balance = {
        "CLL": 4000,
        "MBL": 2000,
        "MCL": 1000,
        "PL": 1000,
        "LPL": 1000,
        "MZL": 1000,
        "FL": 1000,
        "HCL": 1000,
        "normal": 6000,
    }
    mapping = None
    train, validate = load_datasets(data, groups=groups, mapping=mapping, balance=balance)

    som_config = io_functions.load_json(data + "_config.json")

    config = classifier.SOMClassifierConfig(**{
        "tubes": {tube: som_config[tube] for tube in tubes},
        "groups": groups,
        "pad_width": 2,
        "mapping": mapping,
        "cost_matrix": None,
    })
    model = classifier.SOMClassifier(config)
    model.create_model(classifier.create_model_multi_input)
    train, validate = model.create_sequence(train, 32), model.create_sequence(validate, 128)
    model.train_generator(train, validate, epochs=20, class_weight=None)

    model.save(output)
    model.save_information(output)
