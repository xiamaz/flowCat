"""Base reporting objects."""
from enum import Enum
from .file_utils import load_experiments, load_metadata


class ExpType(Enum):
    """Type of experiment."""
    CLUST = 1
    CLASS = 2


class Reporter:
    def __init__(
            self,
            path_classification="output/classification",
            path_clustering="output/clustering"
    ):
        self._path_classification = path_classification
        self._path_clustering = path_clustering

        self._clustering_files = None
        self._classification_files = None

    @property
    def clustering_files(self):
        if self._clustering_files is None:
            data = load_experiments(self._path_clustering)
            data["exptype"] = ExpType.CLUST
            self._clustering_files = data
        return self._clustering_files

    @property
    def classification_files(self):
        if self._classification_files is None:
            data = load_experiments(
                self._path_classification
            )
            data["exptype"] = ExpType.CLASS
            self._classification_files = data
        return self._classification_files

    def get_metadata(self, experiment_row):
        if experiment_row["exptype"] != ExpType.CLASS:
            raise TypeError("Experiment is not a classification experiment")
        return load_metadata(experiment_row["path"])

    def get_experiment_sets(self, experiments):
        """Get experiment groups, that should be processed together."""
        print(experiments.columns)
        exp_sets = list(experiments.groupby(by=["set", "name", "type"]))
        return exp_sets
        
