# MIT License
#
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
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from tensorflow.python.client import device_lib
from ..utils import create_stamp

"""
Modified from code by Chris Gorman.

Adapted from code by Sachin Joglekar
https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
"""

LOGGER = logging.getLogger(__name__)


def create_marker_generator(data_iterable, channels, batch_size=1):
    def marker_generator():
        buf = []
        cache = []

        for i, data in enumerate(data_iterable):
            if i % batch_size == 0 and buf:
                concat = np.concatenate(buf, axis=0)
                yield concat
                cache.append(concat)
                buf = []
            mdata = data[channels].values
            buf.append(mdata)
        if buf:
            concat = np.concatenate(buf, axis=0)
            yield concat
            cache.append(concat)

        while True:
            for cached in cache:
                yield cached

    return marker_generator


def linear_cooling(initial, end, epoch, max_epochs):
    """Implement linear decay of parameter depending on the current epoch."""
    result = tf.subtract(
        initial,
        tf.multiply(
            epoch,
            tf.divide(
                tf.cast(tf.subtract(initial, end), tf.float32),
                tf.cast(tf.subtract(max_epochs, 1), tf.float32))))
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
            softmax_activity=False, output_sensitivity=-1.0,
            initialization_method="sample", reference=None, max_random=1.0,
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
        self._channels = list(channels)
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
        self._softmax_activity = bool(softmax_activity)
        self._model_name = str(model_name)

        if output_sensitivity > 0:
            output_sensitivity *= -1
        elif output_sensitivity == 0:
            output_sensitivity = -1

        # The activity equation is kind of long so I'm naming this c for
        # brevity
        self._c = float(output_sensitivity)

        self._initialization_method = initialization_method

        self._trained = False

        # always run with the maximum number of gpus
        # limit to the first gpu at first
        self._gpus = [d.name for d in device_lib.list_local_devices() if d.device_type == "GPU"]

        # Initialized later, just declaring up here for neatness and to avoid
        # warnings
        self._iter_input = None
        self._weights = None
        self._location_vects = None
        self._input = None
        self._epoch = None
        self._training_op = None
        self._centroid_grid = None
        self._locations = None
        self._activity_op = None
        self._activity_merged = None

        # prediction variables
        self._invar = None
        self._prediction_input = None
        self._squared_distances = None
        self._prediction_output = None
        self._prediction_distance = None
        self._transform_output = None

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

        self._saver = None
        self._summary_list = list()

        # optional for alternative initialization
        self._init_samples = None

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

    def save(self, location):
        """Save the current model into the specified location."""
        tf.saved_model.simple_save(
            self._sess, location,
            inputs={
                "indata": self._invar
            },
            outputs={
                "mapping": self._prediction_output,
                "histogram": self._transform_output,
            },
        )

    def _initialize_tf_graph(self, *args, **kwargs):
        """ Initialize the SOM on the TensorFlow graph

        In multi-gpu mode it will duplicate the model across the GPUs and use
        the CPU to calculate the final weight updates.
        """
        # ensure that input has been provided before initialization
        assert self._input_tensor is not None, "Load input before initializing"

        with self._graph.as_default(), \
                tf.variable_scope(tf.get_variable_scope()), \
                tf.device('/cpu:0'):
            # This list will contain the handles to the numerator and
            # denominator tensors for each of the towers
            tower_updates = list()
            # This is used by all of the towers and needs to be fed to the
            # graph, so let's put it here
            with tf.name_scope('Iteration'):
                self._iter_input = tf.placeholder("float", [], name="iter")

            # always use the first gpu
            if self._gpus:
                with tf.device(self._gpus[0]), tf.name_scope('Tower_0'):
                    # Create the model on this tower and add the
                    # (numerator, denominator) tensors to the list
                    tower_updates.append(self._tower_som(*args, **kwargs))
                    tf.get_variable_scope().reuse_variables()

                    # Put the activity op on the last GPU
                    # self._activity_op = self._make_activity_op(self._input)
            else:
                # Running CPU only
                with tf.name_scope("Tower_0"):
                    tower_updates.append(self._tower_som())
                    tf.get_variable_scope().reuse_variables()

                    # self._activity_op = self._make_activity_op(self._input)

            with tf.name_scope("Weight_Update"):
                # Get the outputs
                numerators, denominators = zip(*tower_updates)
                # Add them up
                numerators = tf.reduce_sum(tf.stack(numerators), axis=0)
                denominators = tf.reduce_sum(tf.stack(denominators), axis=0)
                # Divide them
                new_weights = tf.divide(numerators, denominators)
                # diff new and old weights
                if self._tensorboard:
                    with tf.name_scope("WeightChange"):
                        diff_weights = tf.reshape(
                            tf.sqrt(tf.reduce_sum(tf.pow(self._weights - new_weights, 2), axis=1)),
                            shape=(1, self._m, self._n, 1))
                        # self._activity_op = diff_weights
                        self._summary_list.append(tf.summary.image("WeightDiff", diff_weights))
                # Assign them
                self._training_op = tf.assign(self._weights, new_weights)

        with self._graph.as_default():
            self._prediction_variables()

            # merge all summaries
            self._merged = tf.summary.merge(self._summary_list)

            if self._activity_op is None:
                self._activity_op = tf.no_op()

    def _prediction_variables(self):
        """Create prediction ops"""
        with tf.name_scope("Prediction"):
            self._invar = tf.placeholder(tf.float32)
            dataset = tf.data.Dataset.from_tensors(self._invar)

            self._prediction_input = dataset.make_initializable_iterator()

            # Get the index of the minimum distance for each input item,
            # shape will be [batch_size],
            self._squared_distances = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self._weights, axis=0),
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

    def _tower_som(self, batches_per_epoch):
        """ Build a single SOM tower on the TensorFlow graph """
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
            elif self._initialization_method in ["sample", "reference"]:
                initializer = self._init_samples
                shape = None
            else:
                raise TypeError("Initialization method not supported.")

            self._weights = tf.get_variable(
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
            self._input = tf.identity(self._input_tensor)

        with tf.name_scope('Epoch'):
            global_step = tf.Variable(-1.0, dtype=tf.float32)
            self._epoch = tf.floor(tf.assign_add(global_step, 1.0) / batches_per_epoch)

        # get best matching units for all events in batch
        with tf.name_scope('BMU_Indices'):
            # squared distance of [batch_size, num_neurons], eg for each event
            # to all neurons
            squared_distance = tf.reduce_sum(
                tf.pow(tf.subtract(tf.expand_dims(self._weights, axis=0),
                                   tf.expand_dims(self._input, axis=1)), 2), 2)

            bmu_indices = tf.argmin(squared_distance, axis=1)

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
                self._epoch, self._max_epochs)
            alpha = apply_cooling(
                self._learning_cooling,
                self._initial_learning_rate, self._end_learning_rate,
                self._epoch, self._max_epochs)

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
                    tf.expand_dims(self._input, axis=1)
                ), axis=0)

            # sum neighborhood function, eg the learn rate of each neuron
            # we divide the batch summed new weights through the neighborhood
            # function sum
            # shape: [batch_size, neurons]
            denominator = tf.expand_dims(
                tf.reduce_sum(learning_rate_op, axis=0) + float(1e-12),
                axis=-1)

        with tf.name_scope('Summary'):
            # All summary ops are added to a list and then the merge() function is called at the end of
            # this method
            # learning parameters
            _, update_mean_alpha = tf.metrics.mean(alpha)
            _, update_mean_radius = tf.metrics.mean(radius)

            self._summary_list.append(tf.summary.scalar('alpha', update_mean_alpha))
            self._summary_list.append(tf.summary.scalar('radius', update_mean_radius))

            mean_distance = tf.sqrt(tf.cast(tf.reduce_min(squared_distance, axis=1), tf.float32))
            _, update_mean_dist = tf.metrics.mean(mean_distance)
            self._summary_list.append(tf.summary.scalar('quantization_error', update_mean_dist))

            # proportion of events where 1st and 2nd bmu are not adjacent
            _, top2_indices = tf.nn.top_k(tf.negative(squared_distance), k=2)
            top2_locs = tf.gather(self._location_vects, top2_indices)
            distances = tf.reduce_sum(tf.pow(tf.subtract(top2_locs[:, 0, :], top2_locs[:, 1, :]), 2), 1)
            topographic_error = tf.divide(
                tf.reduce_sum(tf.cast(distances > 1, tf.float32)),
                tf.cast(tf.size(distances), tf.float32))
            self._summary_list.append(tf.summary.scalar("topographic_error", topographic_error))

            # self._activity_op = learning_rate_op
            learn_image = tf.reshape(
                tf.reduce_mean(learning_rate_op, axis=0), shape=(1, self._m, self._n, 1))
            self._summary_list.append(tf.summary.image("learn_img", learn_image))

        with tf.name_scope("WeightsSummary"):
            # combined cd45 ss int lin plot using r and g color
            self._create_color_map(["CD45-KrOr", "SS INT LIN", None], "cd45_ss")
            self._create_color_map([None, "SS INT LIN", "CD19-APCA750"], "ss_cd19")
            if "Kappa-FITC" in self._channels:
                self._create_color_map([None, "Kappa-FITC", "Lambda-PE"], "kappa_lambda")
            self._create_color_map(["CD45-KrOr", "SS INT LIN", "CD19-APCA750"], "zz_cd45_ss_cd19")

        with tf.name_scope("MappingSummary"):
            mapped_events_per_node = tf.reduce_sum(
                tf.one_hot(bmu_indices, self._m * self._n), axis=0)

            event_image = tf.reshape(mapped_events_per_node, shape=(1, self._m, self._n, 1))
            self._summary_list.append(tf.summary.image("mapping_img", event_image))

        return numerator, denominator


    def _create_color_map(self, channels, name="colormap"):
        """Create a color map using given channels. Also generate a small reference visualizing the
        given colorspace."""
        slices = [
            tf.zeros(self._m * self._n) if channel is None else self._weights[:, self._channels.index(channel)]
            for channel in channels
        ]
        marker_image = tf.reshape(tf.stack(slices, axis=1), shape=(1, self._m, self._n, 3))
        self._summary_list.append(tf.summary.image(name, marker_image))

        if None in channels:
            none_pos = channels.index(None)
            legend_list = [[i, j] for i in range(2) for j in range(2)]
            for leg in legend_list:
                leg.insert(none_pos, 0)

            legend_axis = tf.reshape(tf.constant(legend_list, dtype=tf.float16), shape=(1, 2, 2, 3))
            self._summary_list.append(tf.summary.image(f"{name}_legend", legend_axis))


    def _make_activity_op(self, input_tensor):
        """ Creates the op for calculating the activity of a SOM
        :param input_tensor: A tensor to calculate the activity of. Must be of
                shape `[batch_size, dim]` where `dim` is the dimensionality of
                the SOM's weights.
        :return A handle to the newly created activity op:
        """
        with tf.name_scope("Activity"):
            # This constant controls the width of the gaussian.
            # The closer to 0 it is, the wider it is.
            c = tf.constant(self._c, dtype="float32")
            # Get the euclidean distance between each neuron and the input
            # vectors
            dist = tf.norm(
                tf.subtract(
                    tf.expand_dims(self._weights, axis=0),
                    tf.expand_dims(input_tensor, axis=1)),
                name="Distance", axis=-1)  # [batch_size, neurons]
            # squared_distance = tf.reduce_sum(
            #     tf.pow(tf.subtract(tf.expand_dims(self._weights, axis=0),
            #                        tf.expand_dims(self._input, axis=1)),
            #            2), 2)

            # Calculate the Gaussian of the activity. Units with distances
            # closer to 0 will have activities closer to 1.
            activity = tf.exp(
                tf.multiply(
                    tf.pow(
                        dist,
                        2),
                    c), name="Gaussian")

            # Convert the activity into a softmax probability distribution
            if self._softmax_activity:
                activity = tf.divide(
                    tf.exp(
                        activity),
                    tf.expand_dims(tf.reduce_sum(
                            tf.exp(
                                activity), axis=1), axis=-1), name="Softmax")

            output = tf.identity(activity, name="Output")
            return output

    def train(self, data_iterable, num_inputs, step_offset=0):
        """ Train the network on the data provided by the input tensor.
        :param num_inputs: The total number of inputs in the data-set. Used to
                            determine batches per epoch
        :param step_offset: The offset for the global step variable so I don't
                            accidentally overwrite my summaries
        """

        # Build the TensorFlow dataset pipeline per the standard tutorial.
        if self._trained:
            LOGGER.warning("Model is already trained.")

        marker_generator = create_marker_generator(data_iterable, self._channels, batch_size=self._batch_size)

        # Divide by num_gpus to avoid accidentally training on the same data a
        # bunch of times
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

            if self._initialization_method == "sample":
                samples = self._reference.values[np.random.choice(
                    self._reference.shape[0], self._m * self._n, replace=False
                ), :]
                self._init_samples = tf.convert_to_tensor(
                    samples, dtype=tf.float32
                )
            # load initial weights from given reference weights
            elif self._initialization_method == "reference":
                self._init_samples = tf.convert_to_tensor(
                    self._reference.values, dtype=tf.float32
                )

            dataset = dataset.repeat()
            # dataset = dataset.batch(self._batch_size)
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

        global_step = step_offset

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
                # current_batch = batch + (batches_per_epoch * epoch)
                # percent_complete = current_batch / total_batches
                # LOGGER.debug(
                #     "\tBatch %d/%d - %.2f%% complete",
                #     batch,
                #     batches_per_epoch,
                #     percent_complete * 100
                # )

                # if recording summaries; initialize a run while recording, save after batch is done
                if self._tensorboard:
                    summary, _, activity, = self._sess.run(
                        [self._merged, self._training_op, self._activity_op],
                        options=run_options, run_metadata=run_metadata
                    )
                    if activity is not None:
                        print(activity)
                        print(activity.shape)
                else:
                    self._sess.run(
                        self._training_op,
                        feed_dict={
                            self._epoch: epoch}
                    )

                # save the summary if it has been tracked
                if self._tensorboard:
                    self._writer.add_run_metadata(run_metadata, f"step_{global_step}")
                    self._writer.add_summary(summary, global_step)

        self._trained = True
        return self

    @property
    def output_weights(self):
        """
        :return: The weights of the trained SOM as a NumPy array, or `None`
                    if the SOM hasn't been trained
        """
        if self._trained:
            return np.array(self._sess.run(self._weights))

        return None

    @property
    def prediction_input(self):
        """Get the prediction input."""
        return self._prediction_input

    @prediction_input.setter
    def prediction_input(self, value):
        valuearr = value[self._channels].values
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
    """
    Create SOM from input data and transform into the weights
    for each SOM-Node, effectively reducing the data to num_nodes x channels
    dimensions.
    """

    def __init__(self, counts=False, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

        self._model = None
        self._counts = counts
        self.history = []

    def fit(self, X, *_):
        """Always retrain model, if fit is called."""
        self._model = TFSom(*self._args, **self._kwargs)

        self._model.train([X], num_inputs=1)
        return self

    def predict(self, X, *_):
        return self._model.map_to_nodes(X)

    def transform(self, X, *_):
        weights = pd.DataFrame(
            self._model.output_weights, columns=X.columns
        )
        if self._counts:
            weights["counts"] = self._model.map_to_histogram_distribution(
                X, relative=False).tolist()
        return weights
