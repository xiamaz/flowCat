# MIT License
#
# Copyright (c) 2018 Max Zhao
# Copyright (c) 2018 Chris Gorman
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =================================================================================
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

import tensorflow as tf
from flowcat.utils import create_stamp, URLPath

"""
Adapted from code by Chris Gorman.
https://github.com/cgorman/tensorflow-som

Adapted from code by Sachin Joglekar
https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
"""

tf.logging.set_verbosity(tf.logging.WARN)

LOGGER = logging.getLogger(__name__)


def linear_cooling(initial, end, epoch, max_epochs):
    """Implement linear decay of parameter depending on the current epoch."""
    result = tf.subtract(
        tf.cast(initial, tf.float32),
        tf.multiply(
            tf.cast(epoch, tf.float32),
            tf.divide(
                tf.subtract(
                    tf.cast(initial, tf.float32),
                    tf.cast(end, tf.float32)),
                tf.subtract(
                    tf.cast(max_epochs, tf.float32),
                    1.0))))
    return result


def exponential_cooling(initial, end, epoch, max_epochs):
    """Implementation of exponential decay for parameter depending on epoch."""
    # Original from somuclu:
    # if (end == 0.0) {
    #     diff = -log(0.1) / nEpoch;
    # }
    # else {
    #     diff = -log(end / start) / nEpoch;
    # }
    # return start * exp(-epoch * diff);
    if end == 0:
        diff_a = tf.log(0.1)
    else:
        diff_a = tf.log(
            tf.divide(
                tf.cast(end, tf.float32),
                tf.cast(initial, tf.float32)))

    diff = tf.divide(
        diff_a,
        tf.cast(max_epochs, tf.float32))

    result = tf.multiply(
        tf.cast(initial, tf.float32),
        tf.exp(
            tf.multiply(
                tf.cast(epoch, tf.float32),
                diff)))

    return result


def apply_cooling(cooling_type, *args, **kwargs):
    """Wrapper around different cooling functions."""
    if cooling_type == "linear":
        cool_op = linear_cooling(*args, **kwargs)
    elif cooling_type == "exponential":
        cool_op = exponential_cooling(*args, **kwargs)
    else:
        raise TypeError(f"Unknown cooling type: {cooling_type}")
    return cool_op


def planar_distance(matched, locations, *_, **__):
    return tf.subtract(
        tf.expand_dims(locations, axis=0),
        tf.expand_dims(matched, axis=1))


def toroid_distance(matched, locations, map_size, *_, **__):
    abs_subtracted = tf.abs(planar_distance(matched, locations))
    # subtract abs distance from map size
    map_subtracted = tf.subtract(map_size, abs_subtracted)
    # select the smaller from abs and subtracted distance
    distance = tf.minimum(abs_subtracted, map_subtracted)
    return distance


def squared_euclidean_distance(distances):
    """dist = sum((a-b)^2)"""
    euclidean = tf.reduce_sum(tf.pow(distances, 2), axis=2)
    return euclidean


def manhattan_distance(distances):
    """dist = sum(abs(a-b))"""
    manhattan = tf.reduce_sum(tf.abs(distances), axis=2)
    return manhattan


def chebyshev_distance(distances):
    """dist = max(abs(a-b))"""
    chebyshev = tf.reduce_max(tf.abs(distances), axis=2)
    return chebyshev


def calculate_node_distance(matched_location, location_vectors, map_type, distance_type, map_size):
    """Calculate the distance between a list of selected node coordinates and all nodes in the map."""
    if map_type == "planar":
        distance = planar_distance(matched_location, location_vectors, map_size)
    elif map_type == "toroid":
        distance = toroid_distance(matched_location, location_vectors, map_size)
    else:
        raise TypeError(f"Unknown map type: {map_type}")

    if distance_type == "euclidean":
        bmu_distances = squared_euclidean_distance(distance)
    elif distance_type == "manhattan":
        bmu_distances = manhattan_distance(distance)
    elif distance_type == "chebyshev":
        bmu_distances = chebyshev_distance(distance)
    else:
        raise TypeError(f"Unknown distance type: {distance_type}")
    return bmu_distances


def create_initializer(init, init_data, dims):
    """Create initializer for weights.

    Args:
        init - Init method name
        init_data - Additional data for method
        dims - Tuple of (m, n, dim)
    Returns:
        Tuple of initializer and shape for weight initialization
    """
    m, n, dim = dims
    shape = None
    if init == "random":
        initializer = tf.random_uniform_initializer(maxval=init_data)
        shape = [m * n, dim]
    elif init == "reference":
        if isinstance(init_data, pd.DataFrame):
            init_data = init_data.values
        initializer = tf.convert_to_tensor(init_data, dtype=tf.float32)
    elif init == "sample":
        samples = init_data.values[np.random.choice(
            init_data.shape[0], m * n, replace=False
        ), :]
        initializer = tf.convert_to_tensor(
            samples, dtype=tf.float32
        )
    else:
        raise TypeError(init)
    return initializer, shape


def summary_quantization_error(squared_distance):
    """Create quantization error."""
    mean_distance = tf.sqrt(tf.cast(tf.reduce_min(squared_distance, axis=1), tf.float32))
    _, update_mean_dist = tf.metrics.mean(mean_distance)
    return tf.summary.scalar('quantization_error', update_mean_dist)


def summary_topographic_error(squared_distance, location_vects):
    """Generate topographic error."""
    _, top2_indices = tf.nn.top_k(tf.negative(squared_distance), k=2)
    top2_locs = tf.gather(location_vects, top2_indices)
    distances = tf.reduce_sum(tf.pow(tf.subtract(top2_locs[:, 0, :], top2_locs[:, 1, :]), 2), 1)
    topographic_error = tf.divide(
        tf.reduce_sum(tf.cast(distances > 1, tf.float32)),
        tf.cast(tf.size(distances), tf.float32))
    return tf.summary.scalar("topographic_error", topographic_error)


def summary_learning_image(learning_rate, m, n):
    """Create image visualization of learning rate across nodes."""
    learn_image = tf.reshape(
        tf.reduce_mean(learning_rate, axis=0), shape=(1, m, n, 1))
    return tf.summary.image("learn_img", learn_image)


class TFSom:
    """Tensorflow Model of a self-organizing map, without assumptions about
    usage.
    2-D rectangular grid planar Self-Organizing Map with Gaussian neighbourhood
    function.
    """

    def __init__(
            self,
            dims, initialization=None, graph=None,
            max_epochs=10, batch_size=10000, buffer_size=1_000_000,
            initial_radius=None, end_radius=None, radius_cooling="linear",
            initial_learning_rate=0.05, end_learning_rate=0.01, learning_cooling="linear",
            node_distance="euclidean", map_type="planar", std_coeff=0.5,
            subsample_size=None,
            model_name="Self-Organizing-Map",
            tensorboard_dir=None, seed=None,
    ):
        """
        Initialize a self-organizing map on the tensorflow graph
        Args:
            dims: Number of rows and columns and dims per node.
            max_epochs: Number of epochs in training.
            batch_size: Number of cases in a single batch. (Not the number of
                rows in one FCS files, this is more akin to passing multiple FCS
                files to a single training step.)
            initial_radius: Initial radius of neighborhood function.
            end_radius: End radius of neighborhood function on the last epoch.
            radius_cooling: Decay of radius over epochs.
            initial_learning_rate: Learning rate at epoch 0.
            end_learning_rate: Learning rate at last epoch.
            learning_cooling: Decay type of learning rate over epochs.
            node_distance: Distance metric between nodes on the SOM map.
            map_type: Behavior of map edges. Either toroid (wrap-around) or planar (no wrap).
            std_coeff: Coefficient of the neighborhood function.
            subsample_size: Size of subsamples from a single batch. None or a negative value will use the entire input
                data.
            model_name: Name of the SOM model. Used for tensorboard directory names.
            tensorboard_dir: Directory to save tensorboard data to. If none, tensorboard will not be generated.
        """
        # snapshot all local variables for config saving
        config = {k: v for k, v in locals().items() if k != "self"}

        self._m, self._n, self._dim = dims

        self._initial_radius = max(self._m, self._n) / 2.0 if initial_radius is None else float(initial_radius)
        self._end_radius = 1.0 if end_radius is None else float(end_radius)
        self._radius_cooling = radius_cooling

        self._initial_learning_rate = initial_learning_rate
        self._end_learning_rate = end_learning_rate
        self._learning_cooling = learning_cooling

        # node distance calculation option on the SOM map
        self._node_distance = node_distance
        self._map_type = map_type
        self._std_coeff = abs(float(std_coeff))

        self._max_epochs = abs(int(max_epochs))
        self._batch_size = abs(int(batch_size))
        self._model_name = str(model_name)
        self._buffer_size = buffer_size

        # Initialized later, just declaring up here for neatness and to avoid
        # warnings
        self._weights = None
        self._ref_weights = None
        self._epoch = None

        self._training_op = None
        self._assign_trained_op = None
        self._reset_weights_op = None

        # prediction variables
        self._invar = None
        self.__prediction_input = None
        self._squared_distances = None
        self._prediction_output = None
        self._prediction_distance = None
        self._transform_output = None

        self._subsample_size = subsample_size

        self._initialized = False

        # tensorboard visualizations
        if tensorboard_dir:
            self._tensorboard_dir = tensorboard_dir / self.config_name
        else:
            self._tensorboard_dir = None

        if self.tensorboard:
            # save model configuration
            with (self._tensorboard_dir / "config.json").open("w") as f:
                f.writelines(str(config))

        self._summary_list = []

        # This will be the collection of summaries for this subgraph. Add new
        # summaries to it and pass it to merge()
        if graph is None:
            self._graph = tf.Graph()
            assert initialization is None, "Init needs to be on same graph"
            with self._graph.as_default():
                self._initialization = create_initializer("random", 1, (self._m, self._n, self._dim))
        else:
            self._graph = graph
            self._initialization = initialization

        self._seed = seed
        if self._seed is not None:
            LOGGER.info("Setting seed to %d", self._seed)
            with self._graph.as_default():
                tf.set_random_seed(self._seed)

        self._sess = None
        self._writer = None

    @property
    def config_name(self):
        """Create a config string usable as file or directory name."""
        return f"{self._model_name}_{self.config_tag}"

    @property
    def config_tag(self):
        """Create config tag without model name."""
        return f"s{self._m}_e{self._max_epochs}_m{self._map_type}_d{self._node_distance}"

    @property
    def initialized(self):
        return self._initialized

    @property
    def tensorboard(self):
        return self._tensorboard_dir is not None

    @property
    def output_weights(self):
        """
        :return: The weights of the trained SOM as a NumPy array, or `None`
                    if the SOM hasn't been trained
        """
        return np.array(self._sess.run(self._weights))

    @property
    def ref_weights(self):
        return np.array(self._sess.run(self._ref_weights))

    def initialize(self):
        """Initialize the tensorflow graph."""
        if self.initialized:
            raise RuntimeError("Graph already initialized")

        self._sess = tf.Session(
            graph=self._graph,
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,))

        with self._graph.as_default():
            self._data_placeholder = self._initialize_tf_graph()
            self._saver = tf.train.Saver()

            # Initalize all variables
            init_op = tf.global_variables_initializer()
            self._sess.run([init_op])

            # Get some metric variables which we will reset each epoch
            self._metric_initializers = tf.variables_initializer(
                self._graph.get_collection(tf.GraphKeys.METRIC_VARIABLES)
            )

        self._initialized = True
        return self

    def add_summary(self, summary):
        if isinstance(summary, list):
            self._summary_list += summary
        else:
            self._summary_list.append(summary)

    def _create_input(self):
        """Create placeholder inputs for a dataset using an reinitializable iterator."""
        data_placeholder = tf.placeholder(tf.float64)

        dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.shuffle(self._buffer_size, seed=None, reshuffle_each_iteration=True)

        iterator = dataset.make_initializable_iterator()
        input_tensor = iterator.get_next()
        return data_placeholder, iterator, input_tensor

    def _initialize_tf_graph(self):
        """Initialize the SOM on the TensorFlow graph"""
        data, iter_init, input_tensor = self._create_input()
        self._data_initializer = iter_init.initializer

        with tf.variable_scope(tf.get_variable_scope()):
            (
                numerators, denominators,
                self._epoch, self._weights, _, summaries
            ) = self._tower_som(input_tensor, self._initialization)

            self._ref_weights = tf.get_variable(
                name="ref_weights", initializer=self._weights)

            tf.get_variable_scope().reuse_variables()
            self.add_summary(summaries)

            # Divide them
            new_weights = tf.divide(numerators, denominators)
            # diff new and old weights
            if self.tensorboard:
                diff_weights = tf.reshape(
                    tf.sqrt(tf.reduce_sum(tf.pow(self._weights - new_weights, 2), axis=1)),
                    shape=(1, self._m, self._n, 1))
                self.add_summary(tf.summary.image("weight_diff", diff_weights))
                control_deps = [diff_weights]
            else:
                control_deps = []

            # Assign them
            with tf.control_dependencies(control_deps):
                self._training_op = tf.assign(self._weights, new_weights)
            self._assign_trained_op = tf.assign(self._ref_weights, self._weights)
            self._reset_weights_op = tf.assign(self._weights, self._ref_weights)

        return data

    def _tower_som(self, input_tensor, initialization):
        """Build a single SOM tower on the TensorFlow graph
        Args:
            input_tensor: Input event data to be mapped to the SOM should have len(channel) width
            initialization: Given initialization tuple with initializer method and shape.
        Returns:
            (numerator, denominator) describe the weight changes and associated cumulative learn rate per
            node. This can be summed across towers, if we want to parallelize training.
        """
        # Randomly initialized weights for all neurons, stored together
        # as a matrix Variable of shape [num_neurons, input_dims]
        with tf.name_scope('Weights'):
            initializer, shape = initialization

            weights = tf.get_variable(
                name='weights',
                shape=shape,
                initializer=initializer
            )

        # Matrix of size [m*n, 2] for SOM grid locations of neurons.
        # Maps an index to an (x,y) coordinate of a neuron in the map for
        # calculating the neighborhood distance
        location_vects = tf.constant(np.array(
            [[i, j] for i in range(self._m) for j in range(self._n)]
        ), name='Location_Vectors')

        with tf.name_scope('Input'):
            input_copy = tf.cast(input_tensor, tf.float32)
            if self._subsample_size:
                # randomly select samples from the input for training
                random_vals = tf.cast(
                    tf.transpose(tf.expand_dims(
                        tf.random_uniform(
                            (self._subsample_size, )
                        ) * tf.cast(tf.shape(input_copy)[0], tf.float32), axis=0)),
                    tf.int32)
                input_copy = tf.gather_nd(input_copy, random_vals)

        # Feed epoch via feed dict, makes everything much simpler
        with tf.name_scope('Epoch'):
            epoch = tf.placeholder(tf.float32, ())

        # get best matching units for all events in batch
        with tf.name_scope('BMU_Indices'):
            # squared distance of [batch_size, num_neurons], eg for each event
            # to all neurons
            squared_distance = tf.reduce_sum(
                tf.pow(tf.subtract(tf.expand_dims(weights, axis=0),
                                   tf.expand_dims(input_copy, axis=1)), 2), 2)

            bmu_indices = tf.argmin(squared_distance, axis=1, name="map_to_node_index")

            # Dangling operators used to get node event mapping and distances
            tf.sqrt(tf.reduce_min(squared_distance, axis=1), name="min_distance")
            tf.reduce_sum(tf.one_hot(
                bmu_indices, self._m * self._n
            ), 0, name="events_per_node")

            mapped_events_per_node = tf.reduce_sum(
                tf.one_hot(bmu_indices, self._m * self._n), axis=0)

        # get the locations of BMU for each event
        with tf.name_scope('BMU_Locations'):
            bmu_locs = tf.reshape(
                tf.gather(location_vects, bmu_indices), [-1, 2]
            )

        with tf.name_scope('Learning_Rate'):
            # learning rate linearly decreases to 0 at max_epoch
            # α = αi - (epoch / max_epoch * αi)
            # same for radius
            radius = apply_cooling(
                self._radius_cooling,
                self._initial_radius, self._end_radius,
                epoch, self._max_epochs)
            self._radius = radius
            alpha = apply_cooling(
                self._learning_cooling,
                self._initial_learning_rate, self._end_learning_rate,
                epoch, self._max_epochs)

            # calculate the node distances between BMU and all other nodes
            # distance will depend on the used metric and the type of the map
            map_size = tf.constant([self._m, self._n], dtype=tf.int64)
            bmu_distances = calculate_node_distance(
                bmu_locs, location_vects, self._map_type, self._node_distance, map_size)

            # gaussian neighborhood, eg 67% neighborhood with 1std
            # keep in mind, that radius is decreasing with epoch
            neighbourhood_func = tf.exp(
                tf.divide(
                    tf.negative(tf.cast(bmu_distances, "float32")),
                    tf.multiply(
                        tf.square(
                            tf.multiply(
                                radius,
                                self._std_coeff)),
                        2)))

            # learn rate dependent on neighborhood
            learning_rate_op = tf.multiply(neighbourhood_func, alpha)

        with tf.name_scope('Update_Weights'):
            # weight input with learning rate and sum across events, if we
            # divide with the summed learning rate we will get a distance
            # weighted update
            # shape: [num_neurons, dimensions]
            numerator = tf.reduce_sum(
                tf.multiply(
                    tf.expand_dims(learning_rate_op, axis=-1),
                    tf.expand_dims(input_copy, axis=1)
                ), axis=0)

            # sum neighborhood function, eg the learn rate of each neuron
            # we divide the batch summed new weights through the neighborhood
            # function sum
            # shape: [batch_size, neurons]
            denominator = tf.expand_dims(
                tf.reduce_sum(learning_rate_op, axis=0) + float(1e-12),
                axis=-1)

        summaries = []
        if self.tensorboard:
            with tf.name_scope('Summary'):
                _, update_mean_alpha = tf.metrics.mean(alpha)
                summaries.append(tf.summary.scalar('alpha', update_mean_alpha))

                _, update_mean_radius = tf.metrics.mean(radius)
                summaries.append(tf.summary.scalar('radius', update_mean_radius))

                summaries.append(summary_quantization_error(squared_distance))
                summaries.append(summary_topographic_error(squared_distance, location_vects))

                summaries.append(summary_learning_image(learning_rate_op, self._m, self._n))

            with tf.name_scope("MappingSummary"):
                event_image = tf.reshape(mapped_events_per_node, shape=(1, self._m, self._n, 1))
                summaries.append(tf.summary.image("mapping_img", event_image))

        return (
            numerator, denominator, epoch,
            weights, mapped_events_per_node, summaries
        )

    def run_till_tensor(self, tensors, data: np.array, epoch: int):
        assert data.shape[0] <= self._buffer_size, (
            f"Data size {data.shape[0]} > Buffer size {self._buffer_size}. "
            "Samples will be lost on reshuffling. "
            "Increase buffer size to number of samples.")

        self._sess.run(
            [self._data_initializer, self._metric_initializers],
            feed_dict={self._data_placeholder: data, self._epoch: epoch}
        )
        result = self._sess.run(
            tensors,
            feed_dict={self._epoch: epoch}
        )
        return result

    def run_till_op(self, op_name: str, data: np.array, epoch: int):
        operation = self._graph.get_operation_by_name(op_name)
        return self.run_till_tensor(operation.outputs, data, epoch)

    def _run_training(
            self,
            data: np.array,
            set_weights: bool = False,
            label: str = ""):
        """
        Train the SOM for a given number of epochs.

        Args:
            data: Numpy array.
            set_weights: Whether trained weights will be kept after training completes.
        """
        assert data.shape[0] <= self._buffer_size, (
            f"Data size {data.shape[0]} > Buffer size {self._buffer_size}. "
            "Samples will be lost on reshuffling. "
            "Increase buffer size to number of samples.")

        if self.tensorboard:
            # Initialize the summary writer after the session has been initialized
            merged_summaries = tf.summary.merge(self._summary_list)
            self._writer = tf.summary.FileWriter(
                str(self._tensorboard_dir / f"train_{label}_{create_stamp()}"), self._sess.graph)

        LOGGER.info("Training self-organizing Map")
        # reset weights to given values after running
        self._sess.run(self._reset_weights_op)

        global_step = 0
        for epoch in range(self._max_epochs):
            LOGGER.info("Epoch: %d/%d", epoch + 1, self._max_epochs)

            # if the tensorboard flag has been provided (for outputting the summaries)
            if self.tensorboard:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # pylint: disable=no-member
                run_metadata = tf.RunMetadata()

            self._sess.run(
                [self._data_initializer, self._metric_initializers],
                feed_dict={self._data_placeholder: data, self._epoch: epoch}
            )
            while True:
                global_step += 1
                try:
                    LOGGER.info("Global step: %d", global_step)

                    if self.tensorboard:
                        summary, _, = self._sess.run(
                            [merged_summaries, self._training_op],
                            options=run_options, run_metadata=run_metadata,
                            feed_dict={self._epoch: epoch}
                        )
                        self._writer.add_run_metadata(run_metadata, f"step_{global_step}")
                        self._writer.add_summary(summary, global_step)
                    else:
                        self._sess.run(
                            self._training_op,
                            feed_dict={self._epoch: epoch}
                        )

                except tf.errors.OutOfRangeError:
                    break

        # set ref_weights to our current weights, these will be used to reset
        # weights the next time run_training is called.
        if set_weights:
            self._sess.run(self._assign_trained_op)
        return self

    def train(self, data, label="learn") -> TFSom:
        """Train the network on the data provided by the input tensor.
        Args:
            data_iterable: Iterable object returning single pandas dataframes.
        """
        self._run_training(data, set_weights=True, label=label)
        return self

    def transform(self, data, label="transform") -> np.array:
        """Train data using given parameters from initial values transiently."""
        self._run_training(data, set_weights=False, label=label)
        return self._sess.run(self._weights)

    def save(self, path: URLPath):
        """Save the model to the given path. Does not work with buffered readers!"""
        self._saver.save(self._sess, str(path))

    def load(self, path: URLPath):
        """Load model from given path."""
        self._saver.restore(self._sess, str(path))
