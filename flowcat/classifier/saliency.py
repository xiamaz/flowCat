import tensorflow as tf
import keras
import vis.utils as vu
from vis.visualization.saliency import visualize_saliency

from flowcat import utils, io_functions
from . import som_dataset
from .classifier import SOMClassifier


def _calculate_bmu_indexes():
    mapdata = tf.placeholder(tf.float32, shape=(None, None), name="som")
    fcsdata = tf.placeholder(tf.float32, shape=(None, None), name="fcs")
    squared_diffs = tf.pow(tf.subtract(
        tf.expand_dims(mapdata, axis=0),
        tf.expand_dims(fcsdata, axis=1)), 2)
    diffs = tf.reduce_sum(squared_diffs, 2)
    bmu = tf.argmin(diffs, axis=1)
    return bmu, (mapdata, fcsdata)


def bmu_calculator(session):
    calc, args = _calculate_bmu_indexes()

    def model(mapdata, fcsdata):
        return session.run(calc, feed_dict={args[0]: mapdata, args[1]: fcsdata})
    return model


class SOMSaliency(SOMClassifier):
    layer_idx = -1

    @classmethod
    def load(cls, path: utils.URLPath):
        """Load classifier model from the given path."""
        self = super().load(path)
        self.model.layers[-1].activation = keras.activations.linear
        self.model = vu.utils.apply_modifications(self.model)
        return self

    def get_validation_data(self, dataset: som_dataset.SOMDataset) -> som_dataset.SOMDataset:
        return dataset.filter(labels=self.data_ids["validation"])

    def transform(self, case, group, maximization=False):
        """Get saliency gradients for the given group for the selected case."""
        xdata, _ = self.array_from_cases([case])
        input_indices = [*range(len(xdata))]
        gradients = visualize_saliency(
            self.model,
            self.layer_idx,
            self.config.groups.index(group),
            seed_input=xdata,
            input_indices=input_indices,
            maximization=maximization
        )
        return gradients

    def calculate_saliency(self, som_sequence, case, group, maximization=False):
        """Calculates the saliency values / gradients for the case, model and
        each of the classes.
        Args:
            dataset: SOMMapDataset object.
            case: Case object for which the saliency values will be computed.
            group: Select group.
            layer_idx: Index of the layer for which the saleincy values will be
                computed.
            maximization: If true, the maximum of the saliency values over all
                channels will be returned.
        Returns:
            List of gradient values sorted first by tube and then class (e.g.
                [[tube1_class1,tube1_class1][tube2_class1,tube2_class2]]).
        """
        xdata, _ = som_sequence.get_batch_by_label(case.id)
        input_indices = [*range(len(xdata))]
        gradients = visualize_saliency(
            self.model,
            self.layer_idx,
            self.config["groups"].index(group),
            seed_input=xdata,
            input_indices=input_indices,
            maximization=maximization
        )
        return gradients
