"""Manage configuration for different systems."""
import copy

from . import utils


def isinstance_more(obj, otype):
    if otype is None or obj is None:
        return obj is otype
    if isinstance(otype, list) and isinstance(obj, list):
        # check all list members if type is a list
        return all(isinstance(o, otype[0]) for o in obj)
    if isinstance(otype, dict) and isinstance(obj, dict):
        k, v = next(iter(otype.items()))
        return all(isinstance(kk, k) and isinstance(vv, v) for kk, vv in obj.items())
    else:
        return isinstance(obj, otype)


def check_schema(schema, data):
    """Recursively follow dictionaries to check schema.
    Schema mismatches cause errors."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected data to be of dict type")
    data_only_keys = set(data.keys()) - set(schema.keys())
    if data_only_keys:
        raise KeyError(f"Unknown keys {data_only_keys}")

    for key, value in schema.items():
        if isinstance(value, dict):
            # recurse into another schema level
            subdata = data.get(key, {})
            data[key] = check_schema(schema[key], subdata)
        elif isinstance(value, tuple):
            # check type
            valid_types, default = value
            dval = data.get(key, default)
            # None not ok if default None and None not in types
            if not any(isinstance_more(dval, d) for d in valid_types):
                raise TypeError(f"{key} got {type(dval)} not in known types {valid_types}")
            data[key] = dval
    return data


class Config:
    """Create configuration skeletons using provided schemas.

    A schema is a nested dict of tuples with the following format:
    {
        key: ((multiple types, ), default),
        key: ((int,), None), <- this denotes a field that must be specified
        key: (([int],), None), <- this denotes a list of specific type
        key {
            further nestings are possible for dictionaries
        }
    """

    _schema = None
    name = ""
    desc = "Empty configuration. Implement a real configuration by inheriting"
    default = None

    def __init__(self, data):
        self._data = self._check_schema(data)

    @classmethod
    def generate_config(cls, args=None):
        """Create configuration from args and kwargs"""
        raise NotImplementedError

    @classmethod
    def from_file(cls, path):
        """Read configuration from file."""
        if str(path).endswith(".json"):
            data = utils.load_json(path)
        elif str(path).endswith(".toml"):
            data = utils.load_toml(path)
        else:
            raise TypeError(f"Unknown filetype: {path}")
        return cls(data)

    @property
    def data(self):
        return self._data

    def copy(self):
        return self.__class__(copy.deepcopy(self.data))

    def _check_schema(self, data):
        """Check data against the given class schema."""
        if self._schema is None:
            raise NotImplementedError("Config should be created from inherited class with schema.")
        return check_schema(self._schema, data)

    def to_file(self, path):
        """Save configuration to file. Infer format from file ending."""
        if str(path).endswith(".json"):
            utils.save_json(self.data, path)
        elif str(path).endswith(".toml"):
            utils.save_toml(self.data, path)
        else:
            raise TypeError(f"Unknown filetype: {path}")

    @classmethod
    def add_to_arguments(cls, parser, conv=None):
        """Add the configuration to arguments.

        Args:
            parser: ArgumentParser object, to which the argument should be
                added.
            conv: Type convert parser input.
        """
        if conv is None:
            def conv_type(path):
                return cls.from_file(path) if path else cls.generate_config()
        else:
            conv_type = conv

        parser.add_argument(
            f"--{cls.name}",
            help=cls.desc,
            type=conv_type,
            default=cls.default,
        )
        return parser

    def __call__(self, *args):
        """Get data. Provide a variable number of arguments, which will be used
        to traverse the schema correctly."""
        data = self.data
        for arg in args:
            data = data[arg]
        return data

    def __str__(self):
        """Convert to string representation."""
        return utils.to_toml(self.data)

    def __eq__(self, other):
        """Compare if two configurations are equal."""
        return str(self) == str(other)


class PathConfig(Config):
    """Input output paths configuration."""

    name = "pathconfig"
    desc = "Input/Output paths configuration."
    default = "paths.toml"
    _schema = {
        "input": (
            ({str: list},), {
                "FCS": [
                    "/data/flowcat-data/mll-flowdata",
                    "s3://mll-flowdata",
                ],
                "SOM": [
                    "output/mll-sommaps/sample_maps",
                    "s3://mll-sommaps/sample_maps",
                ],
                "HISTO": [
                    "s3://mll-flow-classification/clustering"
                ],
            }
        ),
        "output": (
            ({str: str},), {
                "classification": "output/mll-sommaps/classification",
                "som-reference": "output/mll-sommaps/reference_maps",
                "som-sample": "output/mll-sommaps/sample_maps",
            }
        ),
    }

    @classmethod
    def generate_config(cls, args=None):
        return cls({})


class SOMConfig(Config):
    """SOM generation configuration."""

    name = "somconfig"
    desc = "SOM creation config file."
    default = ""
    _schema = {
        "name": ((str,), None),
        "dataset": {
            "labels": ((str, None), None),
            "names": (({str: str},), {"FCS": "fixedCLL-9F"}),
            "filters": {
                "tubes": (([int],), [1, 2]),
                "num": ((int, None), 1),
                "groups": (([str], None), None),
                "infiltration": ((float, int, None), None),
                "infiltration_max": ((float, int, None), None),
                "counts": ((int, None), 8192),
            },
            "selected_markers": ((dict, None), None),
        },
        "reference": ((str, None), None),
        "randnums": ((dict,), {}),
        "tfsom": {
            "model_name": ((str,), None),
            "m": ((int,), 32),
            "n": ((int,), 32),
            "map_type": ((str,), "toroid"),
            "max_epochs": ((int,), 10),
            "batch_size": ((int,), 1),
            "subsample_size": ((int,), 8192),
            "initial_learning_rate": ((float,), 0.5),
            "end_learning_rate": ((float,), 0.1),
            "learning_cooling": ((str,), "exponential"),
            "initial_radius": ((int,), 16),
            "end_radius": ((int,), 1),
            "radius_cooling": ((str,), "exponential"),
            "node_distance": ((str,), "euclidean"),
            "initialization_method": ((str,), "random"),
        },
        "somnodes": {
            "fitmap_args": ((dict, None), None),
        }
    }

    @classmethod
    def generate_config(cls, args=None):
        return cls({})


class ClassificationConfig(Config):
    """Classification configuration."""

    name = "modelconfig"
    desc = "Classification configuration"
    default = ""
    _schema = {
        "name": ((str,), None),
        "dataset": {
            "names": (({str: str},), {"FCS": "fixedCLL-9F"}),
            "filters": {
                "tubes": (([int],), [1, 2]),
                "counts": ((int, None), None),
                "groups": (([str], None), None),
                "num": ((int, None), None),
            },
            "mapping": ((str, None), None),
        },
        "split": {
            "train_num": ((int, float, None), 0.9),
            "train_labels": ((str, None), None),
            "test_labels": ((str, None), None),
        },
        "model": {
            "type": ((str,), None),
            "loader": {
                "FCSLoader": {
                    "subsample": ((int,), 200),
                    "randomize": ((bool,), False),  # always change subsample in different epochs
                },
                "CountLoader": {
                    "datatype": ((str,), "dataframe"),  # alternative SOM will load counts from 2d maps
                },
                "Map2DLoader": {
                    "sel_count": ((str, None), None),  # whether counts will be included in 2d map
                    "pad_width": ((int,), 0),  # whether map will be padded
                }
            },
            "train_args": {
                "batch_size": ((int,), None),
                "draw_method": ((str,), "random"),  # possible: sequential, shuffle, balanced, groupnum
                "epoch_size": ((int, None), None),
                "sample_weights": ((bool,), False),
            },
            "test_args": ((dict,), {
                "batch_size": ((int,), None),
                "draw_method": ((str,), "sequential"),
                "epoch_size": ((int, None), None),
                "sample_weights": ((bool,), False),
            }),
        },
        "run": {
            "weights": ((str, None), None),
            "train_epochs": ((int,), 100),
            "initial_rate": ((float,), 1e-4),
            "drop": ((float,), 0.5),
            "epochs_drop": ((int,), 50),
            "epsilon": ((float,), 1e-8),
            "num_workers": ((int,), 8),
            "validation": ((bool,), False),
            "pregenerate": ((bool,), True),
        },
        "stat": {
            "confusion_sizes": ((bool,), True),
        },
    }

    @classmethod
    def generate_config(cls, args=None):
        return cls({})
