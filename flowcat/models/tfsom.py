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
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn as sk

import tensorflow as tf
from ..utils import create_stamp
from ..dataset import fcs
from .. import som

"""
Adapted from code by Chris Gorman.
https://github.com/cgorman/tensorflow-som

Adapted from code by Sachin Joglekar
https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
"""

LOGGER = logging.getLogger(__name__)


MARKER_IMAGES = {
    "cd45_ss": ("CD45-KrOr", "SS INT LIN", None),
    "ss_cd19": (None, "SS INT LIN", "CD19-APCA750"),
    "kappa_lambda": (None, "Kappa-FITC", "Lambda-PE"),
    "zz_cd45_ss_cd19": ("CD45-KrOr", "SS INT LIN", "CD19-APCA750"),
}


class MarkerMissingError(Exception):
    def __init__(self, markers, message):
        self.markers = markers
        self.message = message


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


def apply_cooling(cooling_type, *args, **kwargs):
    """Wrapper around different cooling functions."""
    if cooling_type == "linear":
        cool_op = linear_cooling(*args, **kwargs)
    elif cooling_type == "exponential":
        cool_op = exponential_cooling(*args, **kwargs)
    else:
        raise TypeError(f"Unknown cooling type: {cooling_type}")
    return cool_op


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


def create_color_map(weights, cols, name="colormap", img_size=None):
    """Create a color map using given cols. Also generate a small reference visualizing the given colorspace.
    Params:
        weights: Tensor of weights with nodes in rows and marker channels in columns.
        cols: number of columns used for the image. needs to be of length 3
        name: Name of the generated image
        img_size: Size of generated image, otherwise will be inferred as sqrt of weight row count.
    """
    assert len(cols) == 3, "Needs one column for each color, use None to ignore a channel."
    rows = weights.shape[0]
    if img_size is None:
        side_len = int(np.sqrt(rows))
        img_size = (side_len, side_len)

    slices = [
        tf.zeros(weights.shape[0]) if col is None else weights[:, col]
        for col in cols
    ]
    marker_image = tf.reshape(tf.stack(slices, axis=1), shape=(1, *img_size, 3))
    summary_image = tf.summary.image(name, marker_image)
    return summary_image


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
        initializer = tf.convert_to_tensor(init_data.values, dtype=tf.float32)
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
            max_epochs=10, batch_size=1, buffer_size=1_000_000,
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
        self._location_vects = None
        self._global_step = None
        self._epoch = None

        self._global_step_op = None
        self._epoch_op = None
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

        # tensorboard visualizations
        self._tensorboard = tensorboard_dir is not None
        if self._tensorboard:
            self._tensorboard_dir = Path(str(tensorboard_dir)) / self.config_name
            self._tensorboard_dir.mkdir(parents=True, exist_ok=True)
            # save configuration
            with open(str(self._tensorboard_dir / "config.json"), "w") as f:
                f.writelines(str(config))
        else:
            self._tensorboard_dir = None

        self._summary_list = []

        # This will be the collection of summaries for this subgraph. Add new
        # summaries to it and pass it to merge()
        self._input_tensor = None

        if graph is None:
            self._graph = tf.Graph()
            assert initialization is None, "Init needs to be on same graph"
            with self._graph.as_default():
                self._initialization = create_initializer("random", 1, (self._m, self._n, self._dim))
        else:
            self._graph = graph
            self._initialization = initialization

        with self._graph.as_default():
            self._data, self._iter_init, self._input_tensor = self._create_input()
            self._initialize_tf_graph()

        self._seed = seed
        if self._seed is not None:
            LOGGER.info("Setting seed to %d", self._seed)
            random.seed(self._seed)
            with self._graph.as_default():
                tf.set_random_seed(self._seed)

        self._sess = tf.Session(
            graph=self._graph,
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,))

        self._writer = None

    @property
    def config_name(self):
        """Create a config string usable as file or directory name."""
        return f"{self._model_name}_{self.config_tag}"

    @property
    def config_tag(self):
        """Create config tag without model name."""
        return f"s{self._m}_e{self._max_epochs}_m{self._map_type}_d{self._node_distance}"

    def add_summary(self, summary):
        if isinstance(summary, list):
            self._summary_list += summary
        else:
            self._summary_list.append(summary)

    def _create_input(self):
        data_placeholder = tf.placeholder(tf.float64)

        dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.shuffle(self._buffer_size, seed=None, reshuffle_each_iteration=True)

        iterator = dataset.make_initializable_iterator()
        input_tensor = iterator.get_next()
        return data_placeholder, iterator, input_tensor

    def _initialize_tf_graph(self):
        """Initialize the SOM on the TensorFlow graph"""
        with self._graph.as_default(), tf.variable_scope(tf.get_variable_scope()):

            with tf.name_scope("Tower_0"):
                (
                    numerators, denominators,
                    self._global_step, self._global_step_op,
                    self._epoch, self._epoch_op,
                    self._weights, _, summaries
                ) = self._tower_som(input_tensor=self._input_tensor)

                self._ref_weights = tf.get_variable(
                    name="ref_weights", initializer=self._weights)

                tf.get_variable_scope().reuse_variables()
                self.add_summary(summaries)

            with tf.name_scope("Weight_Update"):
                # Divide them
                new_weights = tf.divide(numerators, denominators)
                # diff new and old weights
                if self._tensorboard:
                    with tf.name_scope("WeightChange"):
                        diff_weights = tf.reshape(
                            tf.sqrt(tf.reduce_sum(tf.pow(self._weights - new_weights, 2), axis=1)),
                            shape=(1, self._m, self._n, 1))
                        summaries.append(tf.summary.image("WeightDiff", diff_weights))
                # Assign them
                self._training_op = tf.assign(self._weights, new_weights)
                self._assign_trained_op = tf.assign(self._ref_weights, self._weights)
                self._reset_weights_op = tf.assign(self._weights, self._ref_weights)

        with self._graph.as_default():
            self._prediction_variables(self._weights)

    def _prediction_variables(self, weights):
        """Create prediction ops"""
        with tf.name_scope("Prediction"):
            self._invar = tf.placeholder(tf.float32)
            dataset = tf.data.Dataset.from_tensors(self._invar)

            self.__prediction_input = dataset.make_initializable_iterator()

            # Get the index of the minimum distance for each input item,
            # shape will be [batch_size],
            self._squared_distances = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(weights, axis=0),
                    tf.expand_dims(self.__prediction_input.get_next(), axis=1)
                ), 2), 2
            )
            self._prediction_output = tf.argmin(
                self._squared_distances, axis=1
            )

            # get the minimum distance for each event
            self._prediction_distance = tf.sqrt(tf.reduce_min(
                self._squared_distances, axis=1
            ))

            # Summarize values across columns to get the absolute number
            # of assigned events for each node
            self._transform_output = tf.reduce_sum(tf.one_hot(
                self._prediction_output, self._m * self._n
            ), 0)

    def _tower_som(self, input_tensor):
        """Build a single SOM tower on the TensorFlow graph
        Args:
            input_tensor: Input event data to be mapped to the SOM should have len(channel) width
        Returns:
            (numerator, denominator) describe the weight changes and associated cumulative learn rate per
            node. This can be summed across towers, if we want to parallelize training.
        """
        # Randomly initialized weights for all neurons, stored together
        # as a matrix Variable of shape [num_neurons, input_dims]
        with tf.name_scope('Weights'):
            initializer, shape = self._initialization

            weights = tf.get_variable(
                name='weights',
                shape=shape,
                initializer=initializer
            )

        # Matrix of size [m*n, 2] for SOM grid locations of neurons.
        # Maps an index to an (x,y) coordinate of a neuron in the map for
        # calculating the neighborhood distance
        self._location_vects = tf.constant(np.array(
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

        with tf.name_scope('Epoch'):
            global_step = tf.Variable(0.0, dtype=tf.float32)
            global_step_op = tf.assign_add(global_step, 1.0)
            epoch = tf.Variable(-1.0, dtype=tf.float32)
            epoch_op = tf.assign_add(epoch, 1.0)

        # get best matching units for all events in batch
        with tf.name_scope('BMU_Indices'):
            # squared distance of [batch_size, num_neurons], eg for each event
            # to all neurons
            squared_distance = tf.reduce_sum(
                tf.pow(tf.subtract(tf.expand_dims(weights, axis=0),
                                   tf.expand_dims(input_copy, axis=1)), 2), 2)

            bmu_indices = tf.argmin(squared_distance, axis=1)

            mapped_events_per_node = tf.reduce_sum(
                tf.one_hot(bmu_indices, self._m * self._n), axis=0)

        # get the locations of BMU for each event
        with tf.name_scope('BMU_Locations'):
            bmu_locs = tf.reshape(
                tf.gather(self._location_vects, bmu_indices), [-1, 2]
            )

        with tf.name_scope('Learning_Rate'):
            # learning rate linearly decreases to 0 at max_epoch
            # α = αi - (epoch / max_epoch * αi)
            # same for radius
            radius = apply_cooling(
                self._radius_cooling,
                self._initial_radius, self._end_radius,
                epoch, self._max_epochs)
            alpha = apply_cooling(
                self._learning_cooling,
                self._initial_learning_rate, self._end_learning_rate,
                epoch, self._max_epochs)

            # calculate the node distances between BMU and all other nodes
            # distance will depend on the used metric and the type of the map
            map_size = tf.constant([self._m, self._n], dtype=tf.int64)
            bmu_distances = calculate_node_distance(
                bmu_locs, self._location_vects, self._map_type, self._node_distance, map_size)

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
        if self._tensorboard:
            with tf.name_scope('Summary'):
                _, update_mean_alpha = tf.metrics.mean(alpha)
                summaries.append(tf.summary.scalar('alpha', update_mean_alpha))

                _, update_mean_radius = tf.metrics.mean(radius)
                summaries.append(tf.summary.scalar('radius', update_mean_radius))

                summaries.append(summary_quantization_error(squared_distance))
                summaries.append(summary_topographic_error(squared_distance, self._location_vects))

                summaries.append(summary_learning_image(learning_rate_op, self._m, self._n))

            with tf.name_scope("MappingSummary"):
                event_image = tf.reshape(mapped_events_per_node, shape=(1, self._m, self._n, 1))
                summaries.append(tf.summary.image("mapping_img", event_image))

        return (
            numerator, denominator,
            global_step, global_step_op,
            epoch, epoch_op,
            weights, mapped_events_per_node, summaries
        )

    def fit_map(self, iterable):
        """Map new data to the existing weights. Optionally refit the map to the data.
        Args:
            data_iterable: Iterable container yielding dataframes.
            max_epochs: Train the map for a number of epochs. If < 1, the data will be directly mapped.
            initial_learn, end_learn: Start and end learn rate.
            initial_radius, end_radius: Start and end radius.
        Yields:
            Tuple of node weights and event mapping
        """

        graph = tf.Graph()
        session = tf.Session(graph=graph)
        with graph.as_default():
            # data input as dataset from generator
            dataset = tf.data.Dataset.from_generator(iter(iterable), output_types=tf.float32)
            data_tensor = dataset.make_one_shot_iterator().get_next()
            input_tensor = tf.Variable(data_tensor, validate_shape=False)

            numerator, denominator, global_step, global_step_op, epoch, epoch_op, weights, mapping, summaries = self._tower_som(
                input_tensor=input_tensor
            )
            new_weights = tf.divide(numerator, denominator)
            train_op = tf.assign(weights, new_weights)

            if self._tensorboard:
                summary = tf.summary.merge(summaries)

            var_init = tf.variables_initializer([weights, global_step, epoch, input_tensor])
            metric_init = tf.variables_initializer(graph.get_collection(tf.GraphKeys.METRIC_VARIABLES))
            number = 0
            if self._tensorboard:
                writer = tf.summary.FileWriter(
                    str(self._tensorboard_dir / f"self_{create_stamp()}"), graph)
            while True:
                try:
                    session.run([var_init, metric_init])
                    (event_mapping_prev,) = session.run([mapping])
                    for epoch in range(self._max_epochs):
                        session.run([epoch_op, global_step_op])
                        session.run([train_op])
                    # get final mapping and weights
                    if self._tensorboard:
                        arr_weights, event_mapping, sum_res = session.run([weights, mapping, summary])
                        writer.add_summary(sum_res, number)
                    else:
                        arr_weights, event_mapping = session.run([weights, mapping])

                    # yield the result after training
                    yield arr_weights, event_mapping, event_mapping_prev
                    number += 1
                except tf.errors.OutOfRangeError:
                    break

    def _run_training(self, data, set_weights=False):
        assert data.shape[0] <= self._buffer_size, (
            f"Data size {data.shape[0]} > Buffer size {self._buffer_size}. "
            "Samples will be lost on reshuffling. "
            "Increase buffer size to number of samples.")

        with self._graph.as_default():
            init_op = tf.global_variables_initializer()
            self._sess.run([init_op])

            metric_init = tf.variables_initializer(self._graph.get_collection(tf.GraphKeys.METRIC_VARIABLES))

        if self._tensorboard:
            # Initialize the summary writer after the session has been initialized
            merged_summaries = tf.summary.merge(self._summary_list)
            self._writer = tf.summary.FileWriter(
                str(self._tensorboard_dir / f"train_{create_stamp()}"), self._sess.graph)

        LOGGER.info("Training self-organizing Map")
        if not set_weights:
            self._sess.run(self._reset_weights_op)

        global_step = 0
        for epoch in range(self._max_epochs):
            LOGGER.info("Epoch: %d/%d", epoch + 1, self._max_epochs)

            # if the tensorboard flag has been provided (for outputting the summaries)
            if self._tensorboard:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            # reset metric variables after every batch
            self._sess.run(
                [self._iter_init.initializer, metric_init, self._epoch_op],
                feed_dict={self._data: data}
            )
            # self._sess.run(metric_init)
            while True:
                try:
                    if self._tensorboard:
                        summary, _, = self._sess.run(
                            [merged_summaries, self._training_op],
                            options=run_options, run_metadata=run_metadata
                        )
                    else:
                        self._sess.run(self._training_op)
                    LOGGER.info("Global step: %d", global_step)
                    global_step = int(self._sess.run(self._global_step_op))
                    # save the summary if it has been tracked
                    if self._tensorboard:
                        self._writer.add_run_metadata(run_metadata, f"step_{global_step}")
                        self._writer.add_summary(summary, global_step)
                except tf.errors.OutOfRangeError:
                    if set_weights:
                        self._sess.run(self._assign_trained_op)
                    break
        return self

    def train(self, data):
        """Train the network on the data provided by the input tensor.
        Args:
            data_iterable: Iterable object returning single pandas dataframes.
        """
        self._run_training(data, set_weights=True)
        return self

    def transform(self, data):
        """Train data using given parameters from initial values transiently."""
        self._run_training(data, set_weights=False)
        return self._sess.run(self._weights)

    @property
    def output_weights(self):
        """
        :return: The weights of the trained SOM as a NumPy array, or `None`
                    if the SOM hasn't been trained
        """
        return np.array(self._sess.run(self._weights))

    @property
    def _prediction_input(self):
        """Get the prediction input."""
        return self.__prediction_input

    @_prediction_input.setter
    def _prediction_input(self, value):
        self._sess.run(
            self.__prediction_input.initializer, feed_dict={self._invar: value}
        )

    def map_to_nodes(self, data):
        """Map data to the closest node in the map.
        """
        self._prediction_input = data
        results = []
        while True:
            try:
                res = self._sess.run(
                    self._prediction_output
                )
                results.append(res)
            except tf.errors.OutOfRangeError:
                break
        return np.concatenate(results)

    def map_to_histogram_distribution(self, data, relative=True):
        """Map input data to the distribution across the SOM map.
        Either return absolute values for each node or relative distribution
        across the dataset.

        :param data: Pandas dataframe or np.matrix
        :param relative: Output relative distribution instead of absolute.

        :return: Array of m x n length, eg number of mapped events for each
                    node.
        """
        self._prediction_input = data
        results = np.zeros(self._m * self._n)
        while True:
            try:
                res = self._sess.run(
                    self._transform_output
                )
                results += res
            except tf.errors.OutOfRangeError:
                break

        if relative:
            results = results / np.sum(results)
        return results

    def distance_to_map(self, data):
        """Return the summed loss of the current case."""
        self._prediction_input = data

        # run in batches to get the result
        results = []
        while True:
            try:
                res = self._sess.run(
                    self._prediction_distance
                )
                results.append(res)
            except tf.errors.OutOfRangeError:
                break

        distance = np.concatenate(results)

        avg_distance = np.average(distance)
        # median_distance = np.median(distance)
        return avg_distance


def create_generator(data, randnums=None):
    """Create a generator for the given data. Optionally applying additional transformations.
    Args:
        transforms: Optional transformation pipeline for the data.
        fit_transformer: Fit transformation pipeline to each new sample.
    Returns:
        Tuple of generator function and length of the data.
    """

    def generator_fun():
        for case in data:
            if randnums is not None:
                for i in range(randnums.get(case.parent.group, 1)):
                    yield i, case
            else:
                yield 0, case

    if randnums is not None:
        length = len(list(generator_fun()))
    else:
        length = len(data)

    return generator_fun, length


def create_label_generator(samples, randnums=None):
    datagen, length = create_generator(samples, randnums=randnums)

    def generator_fun():
        for i, sample in datagen():
            yield sample.parent.id, i
    return generator_fun, length


def create_min_max_generator(samples, randnums=None):
    datagen, length = create_generator(samples, randnums=randnums)

    def generator_fun():
        for _, case in datagen():
            yield case.data.scale().data
    return generator_fun, length


def create_z_score_generator(samples, randnums=None, scaler=None):
    """Normalize channel information for mean and standard deviation.
    Args:
        samples: List of samples.
    Returns:
        Generator function and length of samples.
    """
    datagen, length = create_generator(samples, randnums=randnums)

    if scaler is None:
        def generator_fun():
            for _, case in datagen():
                yield case.data.normalize().scale().data
    else:
        def generator_fun():
            for _, case in datagen():
                yield scaler.transform(case.data).data

    return generator_fun, length


def get_generator(name):
    if name == "scale":
        create_generator = create_min_max_generator
    elif name == "zscore":
        create_generator = create_z_score_generator
    else:
        raise KeyError(f"Preprocessing {name} has to be either: scale, zscore")
    return create_generator


class SOMNodes:
    """Transform FCS data into SOM nodes, optionally with number of mapped counts."""

    def __init__(self, counts=False, fitmap_args=None, randnums=None, preprocessing="scale", *args, **kwargs):
        """
        Args:
            counts: Save counts together with marker channel data.
            fitmap_args: Arguments to be passed to each iteration of the fitmap function.
            randnums: Number of random passes for the given group, if the group is not
                in the dict, a single sample will only be assessed once.
        """
        self._model = TFSom(*args, **kwargs)
        self._counts = counts
        self.history = []
        self._fitmap_args = {} if fitmap_args is None else fitmap_args
        self._randnums = {} if randnums is None else randnums
        self._preprocessing = preprocessing

    def fit(self, X, *_):
        """Optionally train the model on the provided data."""
        self._model.train(X)
        return self

    def get_weights(self):
        weights = self._model.output_weights
        weight_df = pd.DataFrame(weights, columns=self._model.channels)
        return weight_df

    def predict(self, X, *_):
        """Return mapping of FCS files to nodes in the model for the given
        tubecase.
        Args:
            X: single tubecase.
        Returns:
            Numpy array of mapping from single fcs events to SOM nodes.
        """
        return self._model.map_to_nodes(X.data.data)

    def transform_generator(self, X, *_):
        """Create a retrained SOM for each single tubecase.
        Args:
            X: Iterable returning single tubecases.
        Yields:
            Dataframe with som node weights and optionally counts.
        """
        datagen, length = get_generator(self._preprocessing)(X, self._randnums)
        labelgen, length = create_label_generator(X, self._randnums)
        labels = list(labelgen())
        for i, (weights, counts, count_prev) in enumerate(
                self._model.fit_map(data_iterable=datagen(), num_inputs=length, **self._fitmap_args)):
            df_weights = pd.DataFrame(weights, columns=self._model.channels)
            if self._counts:
                df_weights["counts"] = counts
                df_weights["count_prev"] = count_prev
            label, randnum = labels[i]
            df_weights.name = f"{label}_{randnum}"
            # df_weights.name = label
            yield label, randnum, df_weights


class FCSSom:
    """Transform FCS data to SOM node weights."""

    def __init__(self, dims, init=("random", None), tube=-1, markers=None, model_name="fcssom", scaler=None, **kwargs):
        self.dims = dims
        m, n, dim = self.dims

        init_type, init_data = init
        if init_type == "random":
            init_data = init_data or 1
        elif init_type == "reference":
            assert isinstance(init_data, som.SOM)
            markers = init_data.markers
            tube = init_data.tube if tube == -1 else tube
            rm, rn = init_data.dims
            init_data = init_data.data
            m = rm if m == -1 else m
            n = rn if n == -1 else n
        elif init_type == "sample":
            assert isinstance(init_data, fcs.FCSData)
            markers = init_data.markers
            init_data = init_data.data

        dim = len(markers) if dim == -1 else dim

        self.tube = tube
        self.name = model_name
        self.markers = list(markers)
        self._graph = tf.Graph()

        with self._graph.as_default():
            initialization = create_initializer(init_type, init_data, (m, n, dim))

        self.model = TFSom(
            (m, n, dim),
            graph=self._graph,
            initialization=initialization,
            model_name=f"{self.name}_t{self.tube}",
            **kwargs)

        with self._graph.as_default():
            self.add_weight_images(MARKER_IMAGES)

        if scaler is None:
            self.scaler = sk.preprocessing.MinMaxScaler()
        else:
            self.scaler = scaler

    def add_weight_images(self, marker_dict):
        """
        Params:
            marker_dict: Dictionary of image name to 3-tuple of markers.
        """
        for name, markers in marker_dict.items():
            try:
                self.add_weight_image(name, markers)
            except MarkerMissingError as m:
                LOGGER.warning("Could not add %s missing %s", name, m.markers)

    def add_weight_image(self, name, markers):
        with tf.name_scope("WeightsSummary"):
            cols = []
            missing = []
            for marker in markers:
                if marker is None:
                    index = None
                else:
                    try:
                        index = self.markers.index(marker)
                    except ValueError:
                        missing.append(marker)
                        continue
                cols.append(index)
            if missing:
                raise MarkerMissingError(missing, "Failed to create weight image")

            color_map = create_color_map(
                self.model._weights, cols,
                name=name, img_size=(*self.dims[:2],))
            self.model.add_summary(color_map)

    @property
    def weights(self):
        data = self.model.output_weights
        dfdata = pd.DataFrame(data, columns=self.markers)
        return som.SOM(dfdata, tube=self.tube)

    def train(self, data, sample=None):
        """Input an iterable with FCSData
        Params:
            data: FCSData object
            sample: Optional subsample to be used in training
        """
        aligned = [d.align(self.markers).data for d in data]
        res = np.concatenate(aligned)
        res = self.scaler.fit_transform(res)

        if sample:
            res = res[np.random.choice(res.shape[0], sample, replace=False), :]

        self.model.train(res)
        return self

    def transform(self, data):
        """Transform input fcs into retrained SOM node weights."""
        aligned = data.align(self.markers).data
        aligned = self.scaler.transform(aligned)
        weights = self.model.transform(aligned)
        somweights = som.SOM(
            pd.DataFrame(weights, columns=self.markers), tube=self.tube)
        return somweights
