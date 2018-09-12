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
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from ..utils import create_stamp

"""
Adapted from code by Chris Gorman.
https://github.com/cgorman/tensorflow-som

Adapted from code by Sachin Joglekar
https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
"""

LOGGER = logging.getLogger(__name__)


def create_marker_generator(data_iterable, channels, batch_size=1, loop=True):
    """Create a generator applying marker channels to provided data and
    optionally batching it to specified larger sizes.
    Args:
        data_iterable: Iterable data, for example a list of dataframes.
        channels: Marker channels used to enforce alignment of data.
        batch_size: Number of individual samples in a single batch If larger than one, multiple samples
            will be concatenated.
        loop: Indefinitely loop the generated data.
    Returns:
        Generator function.
    """

    def marker_generator():
        buf = []
        cache = []

        for i, data in enumerate(data_iterable):
            if i % batch_size == 0 and buf:
                concat = np.concatenate(buf, axis=0)
                yield concat
                if loop:
                    cache.append(concat)
                buf = []
            mdata = data[channels].values
            buf.append(mdata)
        if buf:
            concat = np.concatenate(buf, axis=0)
            yield concat
            if loop:
                cache.append(concat)

        while True and loop:
            # randomize the presentation of the batches, to prevent ecg like
            # fitting to individual samples
            random.shuffle(cache)
            for cached in cache:
                yield cached

    return marker_generator


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
        raise TypeError(f"Unknown distance type: {self._node_distance}")
    return bmu_distances


class TFSom:
    """Tensorflow Model of a self-organizing map, without assumptions about
    usage.
    2-D rectangular grid planar Self-Organizing Map with Gaussian neighbourhood
    function.
    """

    def __init__(
            self,
            m, n, channels,
            max_epochs=10, batch_size=1,
            initial_radius=None, end_radius=None, radius_cooling="linear",
            initial_learning_rate=0.05, end_learning_rate=0.01, learning_cooling="linear",
            node_distance="euclidean", map_type="planar", std_coeff=0.5,
            initialization_method="sample", reference=None, max_random=1.0,
            random_subsample=False,
            model_name="Self-Organizing-Map",
            tensorboard=False, tensorboard_dir="tensorboard"
    ):
        """
        Initialize a self-organizing map on the tensorflow graph
        Args:
            m, n: Number of rows and columns.
            channels: Columns in the input data. Names will be used for
                alignment of the input dataframe prior to training and prediction.
        """
        # snapshot all local variables for config saving
        config = {k: v for k, v in locals().items() if k != "self"}

        self._m = abs(int(m))
        self._n = abs(int(n))
        self.channels = list(channels)
        self._dim = len(channels)

        self._initial_radius = max(m, n) / 2.0 if initial_radius is None else float(initial_radius)
        self._end_radius = 1.0 if end_radius is None else float(end_radius)
        self._radius_cooling = radius_cooling

        self._initial_learning_rate = initial_learning_rate
        self._end_learning_rate = end_learning_rate
        self._learning_cooling = learning_cooling

        # optional reference data that will be used as initial weights or for
        # random sampling as initial node weights
        self._reference = reference
        self._max_random = max_random

        # node distance calculation option on the SOM map
        self._node_distance = node_distance
        self._map_type = map_type
        self._std_coeff = abs(float(std_coeff))

        self._max_epochs = abs(int(max_epochs))
        self._batch_size = abs(int(batch_size))
        self._model_name = str(model_name)

        self._initialization_method = initialization_method

        # Initialized later, just declaring up here for neatness and to avoid
        # warnings
        self._weights = None
        self._location_vects = None
        self._global_step = None
        self._training_op = None
        self._centroid_grid = None
        self._locations = None

        # prediction variables
        self._invar = None
        self._prediction_input = None
        self._squared_distances = None
        self._prediction_output = None
        self._prediction_distance = None
        self._transform_output = None

        self._random_subsample = random_subsample

        # tensorboard visualizations
        self._tensorboard = tensorboard
        if tensorboard:
            self._tensorboard_dir = Path(tensorboard_dir) / self.config_name
            self._tensorboard_dir.mkdir(parents=True, exist_ok=True)
            # save configuration
            with open(str(self._tensorboard_dir / "config.json"), "w") as f:
                f.writelines(str(config))
        else:
            self._tensorboard_dir = None

        self._summary_list = list()

        # This will be the collection of summaries for this subgraph. Add new
        # summaries to it and pass it to merge()
        self._input_tensor = None

        self._graph = tf.Graph()

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

    def _initialize_tf_graph(self, batches_per_epoch):
        """Initialize the SOM on the TensorFlow graph"""
        with self._graph.as_default(), tf.variable_scope(tf.get_variable_scope()):
            with tf.name_scope("Tower_0"):
                numerators, denominators, self._global_step, self._weights, _, summaries = self._tower_som(
                    input_tensor=self._input_tensor,
                    batches_per_epoch=batches_per_epoch,
                    max_epochs=self._max_epochs,
                    initial_radius=self._initial_radius,
                    initial_learning_rate=self._initial_learning_rate,
                    end_radius=self._end_radius,
                    end_learning_rate=self._end_learning_rate,
                )
                tf.get_variable_scope().reuse_variables()

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

        with self._graph.as_default():
            self._prediction_variables(self._weights)
            # merge all summaries
            self._merged = tf.summary.merge(self._summary_list + summaries)

    def _prediction_variables(self, weights,):
        """Create prediction ops"""
        with tf.name_scope("Prediction"):
            self._invar = tf.placeholder(tf.float32)
            dataset = tf.data.Dataset.from_tensors(self._invar)

            self._prediction_input = dataset.make_initializable_iterator()

            # Get the index of the minimum distance for each input item,
            # shape will be [batch_size],
            self._squared_distances = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(weights, axis=0),
                    tf.expand_dims(self._prediction_input.get_next(), axis=1)
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

    def _tower_som(
            self,
            input_tensor,
            batches_per_epoch, max_epochs,
            initial_radius, end_radius,
            initial_learning_rate, end_learning_rate,
    ):
        """Build a single SOM tower on the TensorFlow graph
        Args:
            input_tensor: Input event data to be mapped to the SOM should have len(channel) width
            batches_per_epoch: Number of batches per epoch, needed to correctly decay learn rate and radius
        Returns:
            (numerator, denominator) describe the weight changes and associated cumulative learn rate per
            node. This can be summed across towers, if we want to parallelize training.
        """
        # Randomly initialized weights for all neurons, stored together
        # as a matrix Variable of shape [num_neurons, input_dims]
        with tf.name_scope('Weights'):
            # Each tower will get its own copy of the weights variable. Since
            # the towers are constructed sequentially, the handle to the
            # Tensors will be different for each tower even if we reference
            # "self"
            if self._initialization_method == "random":
                initializer = tf.random_uniform_initializer(maxval=self._max_random)
                shape = [self._m * self._n, self._dim]
            elif self._initialization_method == "sample":
                samples = self._reference.values[np.random.choice(
                    self._reference.shape[0], self._m * self._n, replace=False
                ), :]
                initializer = tf.convert_to_tensor(
                    samples, dtype=tf.float32
                )
                shape = None
            elif self._initialization_method == "reference":
                initializer = tf.convert_to_tensor(
                    self._reference.values, dtype=tf.float32
                )
                shape = None
            else:
                raise TypeError("Initialization method not supported.")

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
            input_copy = tf.identity(input_tensor)
            if self._random_subsample:
                random_vals = tf.cast(
                    tf.transpose(tf.expand_dims(
                        tf.random_uniform((4 * self._m * self._n, )) * tf.cast(tf.shape(input_copy)[0],
                                                                           tf.float32),
                        axis=0)),
                    tf.int32)
                input_copy = tf.gather_nd(input_copy, random_vals)

        with tf.name_scope('Epoch'):
            global_step = tf.Variable(-1.0, dtype=tf.float32)
            epoch = tf.floor(tf.assign_add(global_step, 1.0) / batches_per_epoch)

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
                initial_radius, end_radius,
                epoch, max_epochs)
            alpha = apply_cooling(
                self._learning_cooling,
                initial_learning_rate, end_learning_rate,
                epoch, max_epochs)

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
                # All summary ops are added to a list and then the merge() function is called at the end of
                # this method
                # learning parameters
                _, update_mean_alpha = tf.metrics.mean(alpha)
                _, update_mean_radius = tf.metrics.mean(radius)

                summaries.append(tf.summary.scalar('alpha', update_mean_alpha))
                summaries.append(tf.summary.scalar('radius', update_mean_radius))

                mean_distance = tf.sqrt(tf.cast(tf.reduce_min(squared_distance, axis=1), tf.float32))
                _, update_mean_dist = tf.metrics.mean(mean_distance)
                summaries.append(tf.summary.scalar('quantization_error', update_mean_dist))

                # proportion of events where 1st and 2nd bmu are not adjacent
                _, top2_indices = tf.nn.top_k(tf.negative(squared_distance), k=2)
                top2_locs = tf.gather(self._location_vects, top2_indices)
                distances = tf.reduce_sum(tf.pow(tf.subtract(top2_locs[:, 0, :], top2_locs[:, 1, :]), 2), 1)
                topographic_error = tf.divide(
                    tf.reduce_sum(tf.cast(distances > 1, tf.float32)),
                    tf.cast(tf.size(distances), tf.float32))
                summaries.append(tf.summary.scalar("topographic_error", topographic_error))

                learn_image = tf.reshape(
                    tf.reduce_mean(learning_rate_op, axis=0), shape=(1, self._m, self._n, 1))
                summaries.append(tf.summary.image("learn_img", learn_image))

            with tf.name_scope("WeightsSummary"):
                # combined cd45 ss int lin plot using r and g color
                summaries.append(
                    self._create_color_map(weights, ["CD45-KrOr", "SS INT LIN", None], "cd45_ss"))
                summaries.append(
                    self._create_color_map(weights, [None, "SS INT LIN", "CD19-APCA750"], "ss_cd19"))
                if "Kappa-FITC" in self.channels:
                    summaries.append(
                        self._create_color_map(weights, [None, "Kappa-FITC", "Lambda-PE"], "kappa_lambda"))
                summaries.append(
                    self._create_color_map(weights, ["CD45-KrOr", "SS INT LIN", "CD19-APCA750"], "zz_cd45_ss_cd19"))

            with tf.name_scope("MappingSummary"):
                event_image = tf.reshape(mapped_events_per_node, shape=(1, self._m, self._n, 1))
                summaries.append(tf.summary.image("mapping_img", event_image))

        return numerator, denominator, global_step, weights, mapped_events_per_node, summaries


    def _create_color_map(self, weights, channels, name="colormap"):
        """Create a color map using given channels. Also generate a small reference visualizing the
        given colorspace."""
        slices = [
            tf.zeros(self._m * self._n) if channel is None else weights[:, self.channels.index(channel)]
            for channel in channels
        ]
        marker_image = tf.reshape(tf.stack(slices, axis=1), shape=(1, self._m, self._n, 3))
        summary_image = tf.summary.image(name, marker_image)

        # if None in channels:
        #     none_pos = channels.index(None)
        #     legend_list = [[i, j] for i in range(2) for j in range(2)]
        #     for leg in legend_list:
        #         leg.insert(none_pos, 0)

        #     legend_axis = tf.reshape(tf.constant(legend_list, dtype=tf.float16), shape=(1, 2, 2, 3))
        #     self._summary_list.append(tf.summary.image(f"{name}_legend", legend_axis))
        return summary_image

    def fit_map(
            self, data_iterable,
            max_epochs=0,
            initial_learn=0.1, end_learn=0.01,
            initial_radius=3, end_radius=1
    ):
        """Map new data to the existing weights. Optionally refit the map to the data.
        Args:
            data_iterable: Iterable container yielding dataframes.
            max_epochs: Train the map for a number of epochs. If < 1, the data will be directly mapped.
            initial_learn, end_learn: Start and end learn rate.
            initial_radius, end_radius: Start and end radius.
        Yields:
            Tuple of node weights and event mapping
        """
        marker_generator = create_marker_generator(data_iterable, self.channels, batch_size=1, loop=False)

        graph = tf.Graph()
        session = tf.Session(graph=graph)
        with graph.as_default():
            # data input as dataset from generator
            dataset = tf.data.Dataset.from_generator(
                marker_generator, output_types=tf.float32, output_shapes=(None, len(self.channels)))
            data_tensor = dataset.make_one_shot_iterator().get_next()
            input_tensor = tf.Variable(data_tensor, validate_shape=False)

            numerator, denominator, global_step, weights, mapping, summaries = self._tower_som(
                input_tensor=input_tensor, batches_per_epoch=1, max_epochs=max_epochs,
                initial_radius=initial_radius, end_radius=end_radius,
                initial_learning_rate=initial_learn, end_learning_rate=end_radius,
            )
            new_weights = tf.divide(numerator, denominator)
            train_op = tf.assign(weights, new_weights)

            summary = tf.summary.merge(summaries)

            var_init = tf.variables_initializer([weights, global_step, input_tensor])
            metric_init = tf.variables_initializer(graph.get_collection(tf.GraphKeys.METRIC_VARIABLES))
            number = 0
            if self._tensorboard:
                writer = tf.summary.FileWriter(
                    str(self._tensorboard_dir / f"self_{create_stamp()}"), graph)
            while True:
                try:
                    session.run([var_init, metric_init])
                    for epoch in range(max_epochs):
                        session.run([train_op])
                    # get final mapping and weights
                    if self._tensorboard:
                        arr_weights, event_mapping, sum_res = session.run([weights, mapping, summary])
                        writer.add_summary(sum_res, number)
                    else:
                        arr_weights, event_mapping = session.run([weights, mapping])

                    # yield the result after training
                    yield arr_weights, event_mapping
                    number += 1
                except tf.errors.OutOfRangeError:
                    break

    def train(self, data_iterable, num_inputs):
        """Train the network on the data provided by the input tensor.
        Args:
            data_iterable: Iterable object returning single pandas dataframes.
            num_inputs: The total number of inputs in the data-set. Used to determine batches per epoch
        """
        marker_generator = create_marker_generator(data_iterable, self.channels, batch_size=self._batch_size)

        batches_per_epoch = int(num_inputs / self._batch_size + 0.5)
        total_batches = batches_per_epoch * self._max_epochs

        # initialize the input fitting dataset
        # the fitting input tensor is directly integrated into the
        # graph, which is why graph creation is postponed until fitting,
        # when we know the actual data of our input
        with self._graph.as_default():
            dataset = tf.data.Dataset.from_generator(
                marker_generator, tf.float32
            )
            print("Batching the dataset")

            self._input_tensor = dataset.make_one_shot_iterator().get_next()

            # Create the ops and put them on the graph
            self._initialize_tf_graph(batches_per_epoch)

            init_op = tf.global_variables_initializer()
            self._sess.run([init_op])

            metric_init = tf.variables_initializer(self._graph.get_collection(tf.GraphKeys.METRIC_VARIABLES))

        if self._tensorboard:
            # Initialize the summary writer after the session has been initialized
            self._writer = tf.summary.FileWriter(
                str(self._tensorboard_dir / f"train_{create_stamp()}"), self._sess.graph)

        global_step = 0
        LOGGER.info("Training self-organizing Map")
        for epoch in range(self._max_epochs):
            LOGGER.info("Epoch: %d/%d", epoch + 1, self._max_epochs)

            # if the tensorboard flag has been provided (for outputting the summaries)
            if self._tensorboard:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            # reset metric variables after every batch
            self._sess.run(metric_init)
            for batch in range(batches_per_epoch):
                global_step += 1

                # if recording summaries; initialize a run while recording, save after batch is done
                if self._tensorboard:
                        summary, _, = self._sess.run(
                        [self._merged, self._training_op],
                        options=run_options, run_metadata=run_metadata
                    )
                else:
                    self._sess.run(self._training_op)

                # save the summary if it has been tracked
                if self._tensorboard:
                    self._writer.add_run_metadata(run_metadata, f"step_{global_step}")
                    self._writer.add_summary(summary, global_step)
        return self

    @property
    def output_weights(self):
        """
        :return: The weights of the trained SOM as a NumPy array, or `None`
                    if the SOM hasn't been trained
        """
        return np.array(self._sess.run(self._weights))

    @property
    def prediction_input(self):
        """Get the prediction input."""
        return self._prediction_input

    @prediction_input.setter
    def prediction_input(self, value):
        valuearr = value[self.channels].values
        self._sess.run(
            self._prediction_input.initializer, feed_dict={self._invar: valuearr}
        )

    def map_to_nodes(self, data):
        """Map data to the closest node in the map."""
        self.prediction_input = data
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
        self.prediction_input = data
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
        self.prediction_input = data

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


class SelfOrganizingMap(BaseEstimator, TransformerMixin):
    """SOM abstraction for usage as a scikit learn transformer."""

    def __init__(self, *args, **kwargs):
        """Expose a subset of tested parameters for external tuning."""
        self._model = TFSom(*args, **kwargs)

    @property
    def weights(self):
        """Return the list of weights."""
        weight_df = pd.DataFrame(
            self._model.output_weights, columns=self._columns
        )
        return weight_df

    @classmethod
    def load(cls, path):
        """Load configuration and state from saved configuration."""
        pass

    def save(self, path):
        """Save inner model and add some additional metadata to be saved."""
        self._model.save(path)

    def fit(self, data, *_):
        """Fit the data using a matrix containing the data. The input
        can be either a numpy matrix or a pandas dataframe."""
        self._model.train(data)
        return self

    def predict(self, data):
        """Predict cluster center for each event in the given data.
        :param data: Input data in tensorflow object.
        :return: List of cluster centers for each event.
        """
        return self._model.map_to_nodes(data)

    def transform(self, data):
        """Transform data of individual events to histogram of events per
        cluster center.
        """
        return self._model.map_to_histogram_distribution(data)


class SOMNodes(BaseEstimator, TransformerMixin):
    """Transform FCS data into SOM nodes, optionally with number of mapped counts."""

    def __init__(self, counts=False, fitmap_args=None, *args, **kwargs):
        """
        Args:
            counts: Save counts together with marker channel data.
        """
        self._model = TFSom(*args, **kwargs)
        self._counts = counts
        self.history = []
        self._fitmap_args = {} if fitmap_args is None else fitmap_args

    def fit(self, X, *_):
        """Optionally train the model on the provided data."""
        self._model.train(X, num_inputs=len(X))
        return self

    def transform(self, X, *_):
        for weights, counts in self._model.fit_map(data_iterable=X, **self._fitmap_args):
            df_weights = pd.DataFrame(weights, columns=self._model.channels)
            if self._counts:
                df_weights["counts"] = counts
            yield df_weights
