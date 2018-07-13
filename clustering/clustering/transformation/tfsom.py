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

__author__ = "Chris Gorman"
__email__ = "chris@cgorman.net"

"""
Modified from code by Chris Gorman.

Adapted from code by Sachin Joglekar
https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
"""

LOGGER = logging.getLogger(__name__)


class TFSom:
    """Tensorflow Model of a self-organizing map, without assumptions about
    usage.
    2-D rectangular grid planar Self-Organizing Map with Gaussian neighbourhood
    function.
    """

    def __init__(
            self,
            m, n,
            max_epochs=10,
            batch_size=4096, test_batch_size=8192,
            initial_radius=None,
            initial_learning_rate=0.1,
            std_coeff=0.5,
            softmax_activity=False,
            output_sensitivity=-1.0,
            initialization_method="sample",
            model_name="Self-Organizing-Map"
    ):
        """
        Initialize a self-organizing map on the tensorflow graph
        :param m: Number of rows of neurons
        :param n: Number of columns of neurons
        :param max_epochs: Number of epochs to train for
        :param batch_size: Number of input vectors to train on at a time
        :param initial_radius: Starting value of the neighborhood radius -
                defaults to max(m, n) / 2.0
        :param initial_learning_rate: The starting learning rate of the SOM.
                Decreases linearly w/r/t `max_epochs`
        :param graph: The tensorflow graph to build the network on
        :param std_coeff: Coefficient of the standard deviation of the
                neighborhood function
        :param model_name: The name that will be given to the checkpoint files
        :param softmax_activity: If `True` the activity will be softmaxed to
                form a probability distribution
        :param output_sensitivity: The constant controlling the width of the
                activity gaussian. See the Jupyter Notebook
                for an explanation.
        :param initialization_method: method used to initialize the som nodes.
        Choices are either random number initialization or sample based
        initialization.
        """

        self._m = abs(int(m))
        self._n = abs(int(n))

        if initial_radius is None:
            self._initial_radius = max(m, n) / 2.0
        else:
            self._initial_radius = float(initial_radius)

        self._max_epochs = abs(int(max_epochs))
        self._batch_size = abs(int(batch_size))
        self._test_batch_size = abs(int(test_batch_size))
        self._std_coeff = abs(float(std_coeff))
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
        self._gpus = [
            d.name for d in device_lib.list_local_devices()
            if d.device_type == "GPU"
        ][0:1]

        # Initialized later, just declaring up here for neatness and to avoid
        # warnings
        self._iter_input = None
        self._dim = None  # dimensionality is inferred from fit input
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
        self._prediction_output = None
        self._transform_output = None

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
                log_device_placement=False,
            )
        )

        self._initial_learning_rate = initial_learning_rate

    def _neuron_locations(self):
        """ Maps an absolute neuron index to a 2d vector for calculating the
        neighborhood function """
        for i in range(self._m):
            for j in range(self._n):
                yield np.array([i, j])

    def _initialize_tf_graph(self):
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

            if self._gpus:
                for i, gpu_name in enumerate(self._gpus):
                    with tf.device(gpu_name):
                        with tf.name_scope('Tower_{}'.format(i)):
                            # Create the model on this tower and add the
                            # (numerator, denominator) tensors to the list
                            tower_updates.append(self._tower_som())
                            tf.get_variable_scope().reuse_variables()

                with tf.device(self._gpus[-1]):
                    # Put the activity op on the last GPU
                    self._activity_op = self._make_activity_op(
                        self._input_tensor
                    )
            else:
                # Running CPU only
                with tf.name_scope("Tower_0"):
                    tower_updates.append(self._tower_som())
                    tf.get_variable_scope().reuse_variables()
                    self._activity_op = self._make_activity_op(
                        self._input_tensor
                    )

            with tf.name_scope("Weight_Update"):
                # Get the outputs
                numerators, denominators = zip(*tower_updates)
                # Add them up
                numerators = tf.reduce_sum(tf.stack(numerators), axis=0)
                denominators = tf.reduce_sum(tf.stack(denominators), axis=0)
                # Divide them
                new_weights = tf.divide(numerators, denominators)
                # Assign them
                self._training_op = tf.assign(self._weights, new_weights)

        # use autoplacement until we know how to parallelize across
        # multiple gpus
        with self._graph.as_default():
            self._prediction_variables()

    def _prediction_variables(self):
        """Create prediction ops"""
        with tf.name_scope("Prediction"):
            self._invar = tf.placeholder(tf.float32)
            self._prediction_input = tf.data.Dataset.from_tensor_slices(
                self._invar
            ).batch(
                self._test_batch_size
            ).make_initializable_iterator()

            # Get the index of the minimum distance for each input item,
            # shape will be [batch_size],
            self._prediction_output = tf.argmin(tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self._weights, axis=0),
                    tf.expand_dims(self._prediction_input.get_next(), axis=1)
                ), 2), 2
            ), axis=1)

            # Summarize values across columns to get the absolute number
            # of assigned events for each node
            self._transform_output = tf.reduce_sum(tf.one_hot(
                self._prediction_output, self._m * self._n
            ), 0)

    def _tower_som(self):
        """ Build a single SOM tower on the TensorFlow graph """
        # Randomly initialized weights for all neurons, stored together
        # as a matrix Variable of shape [num_neurons, input_dims]
        with tf.name_scope('Weights'):
            # Each tower will get its own copy of the weights variable. Since
            # the towers are constructed sequentially, the handle to the
            # Tensors will be different for each tower even if we reference
            # "self"
            if self._initialization_method == "random":
                initializer = tf.random_uniform_initializer(maxval=1)
                shape = [self._m * self._n, self._dim]
            elif self._initialization_method == "sample":
                initializer = self._init_samples
                shape = None
            else:
                raise TypeError("Initialization method not supported.")
            self._weights = tf.get_variable(
                name='weights',
                shape=shape,
                initializer=initializer,
            )

        # Matrix of size [m*n, 2] for SOM grid locations of neurons.
        # Maps an index to an (x,y) coordinate of a neuron in the map for
        # calculating the neighborhood distance
        self._location_vects = tf.constant(np.array(
            list(self._neuron_locations())), name='Location_Vectors')

        with tf.name_scope('Input'):
            self._input = tf.identity(self._input_tensor)

        with tf.name_scope('Epoch'):
            self._epoch = tf.placeholder("float", [], name="iter")

        # Start by computing the best matching units / winning units for each
        # input vector in the batch.
        # Basically calculates the Euclidean distance between every neuron's
        # weight vector and the inputs, and returns the index of the neurons
        # which give the least value
        # Since we are doing batch processing of the input, we need to
        # calculate a BMU for each of the individual inputs in the batch. Will
        # have the shape [batch_size]

        # Oh also any time we call expand_dims it's almost always so we can
        # make TF broadcast stuff properly
        with tf.name_scope('BMU_Indices'):
            # Distance between weights and the input vector
            # Note we are reducing along 2nd axis so we end up with a tensor of
            # [batch_size, num_neurons] corresponding to the distance between a
            # particular input and each neuron in the map
            # Also note we are getting the squared distance because there's no
            # point calling sqrt or tf.norm if we're just doing a strict
            # comparison
            squared_distance = tf.reduce_sum(
                tf.pow(tf.subtract(tf.expand_dims(self._weights, axis=0),
                                   tf.expand_dims(self._input, axis=1)), 2), 2)

            # Get the index of the minimum distance for each input item, shape
            # will be [batch_size],
            bmu_indices = tf.argmin(squared_distance, axis=1)

        # This will extract the location of the BMU in the map for each input
        # based on the BMU's indices
        with tf.name_scope('BMU_Locations'):
            # Using tf.gather we can use `bmu_indices` to index the location
            # vectors directly
            bmu_locs = tf.reshape(
                tf.gather(self._location_vects, bmu_indices), [-1, 2]
            )

        with tf.name_scope('Learning_Rate'):
            # With each epoch, the initial sigma value decreases linearly
            radius = tf.subtract(
                self._initial_radius,
                tf.multiply(
                    self._epoch,
                    tf.divide(
                        tf.cast(
                            tf.subtract(self._initial_radius, 1), tf.float32
                        ),
                        tf.cast(
                            tf.subtract(self._max_epochs, 1), tf.float32
                        )
                    )
                )
            )

            alpha = tf.subtract(
                self._initial_learning_rate,
                tf.multiply(
                    self._epoch,
                    tf.divide(
                        tf.cast(
                            tf.subtract(self._initial_learning_rate, 1),
                            tf.float32
                        ),
                        tf.cast(
                            tf.subtract(self._max_epochs, 1),
                            tf.float32
                        )
                    )
                )
            )

            # Construct the op that will generate a matrix with learning rates
            # for all neurons and all inputs, based on iteration number and
            # location to BMU

            # Start by getting the squared difference between each BMU location
            # and every other unit in the map bmu_locs is [batch_size, 2], i.e.
            # the coordinates of the BMU for each input vector.
            # location vects shape should be [1, num_neurons, 2]
            # bmu_locs should be [batch_size, 1, 2]
            # Output needs to be [batch_size, num_neurons], i.e. a row vector
            # of distances for each input item
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                tf.expand_dims(self._location_vects, axis=0),
                tf.expand_dims(bmu_locs, axis=1)), 2), 2)

            # Using the distances between each BMU, construct the Gaussian
            # neighborhood function.
            # Basically, neurons which are close to the winner will move more
            # than those further away.  The radius tensor decreases the width
            # of the Gaussian over time, so early in training more neurons will
            # be affected by the winner and by the end of training only the
            # winner will move.
            # This tensor will be of shape [batch_size, num_neurons] as well
            # and will be the value multiplied to each neuron based on its
            # distance from the BMU for each input vector
            neighbourhood_func = tf.exp(
                tf.divide(
                    tf.negative(
                        tf.cast(bmu_distance_squares, "float32")
                    ),
                    tf.multiply(
                        tf.square(tf.multiply(radius, self._std_coeff)), 2
                    )
                )
            )

            # Finally multiply by the learning rate to decrease overall neuron
            # movement over time
            learning_rate_op = tf.multiply(neighbourhood_func, alpha)

        # The batch formula for SOMs multiplies a neuron's neighborhood by all
        # of the input vectors in the batch, then divides that by just the sum
        # of the neighborhood function for each of the inputs.
        # We are writing this in a way that performs that operation for each of
        # the neurons in the map.
        with tf.name_scope('Update_Weights'):
            # The numerator needs to be shaped [num_neurons, dimensions] to
            # represent the new weights for each of the neurons. At this point,
            # the learning rate tensor will be shaped [batch_size, neurons].
            # The end result is that, for each neuron in the network, we use
            # the learning rate between it and each of the input vectors, to
            # calculate a new set of weights.
            numerator = tf.reduce_sum(
                tf.multiply(
                    tf.expand_dims(learning_rate_op, axis=-1),
                    tf.expand_dims(self._input, axis=1)
                ), axis=0
            )

            # The denominator is just the sum of the neighborhood functions for
            # each neuron, so we get the sum along axis 1 giving us an output
            # shape of [num_neurons]. We then expand the dims so we can
            # broadcast for the division op. Again we transpose the learning
            # rate tensor so it's [num_neurons, batch_size] representing the
            # learning rate of each neuron for each input vector
            denominator = tf.expand_dims(
                tf.reduce_sum(learning_rate_op, axis=0) + float(1e-12),
                axis=-1
            )

        # We on;y really care about summaries from one of the tower SOMs, so
        # assign the merge op to the last tower we make. Otherwise there's way
        # too many on Tensorboard.
        # self._merged = tf.summary.merge(self._summary_list)

        # With multi-gpu training we collect the results and do the weight
        # assignment on the CPU
        return numerator, denominator

    def _make_activity_op(self, input_tensor):
        """ Creates the op for calculating the activity of a SOM
        :param input_tensor: A tensor to calculate the activity of. Must be of
                shape `[batch_size, dim]` where `dim` is the dimensionality of
                the SOM's weights.
        :return A handle to the newly created activity op:
        """
        with self._graph.as_default():
            with tf.name_scope("Activity"):
                # This constant controls the width of the gaussian.
                # The closer to 0 it is, the wider it is.
                c = tf.constant(self._c, dtype="float32")
                # Get the euclidean distance between each neuron and the input
                # vectors
                dist = tf.norm(tf.subtract(
                    tf.expand_dims(self._weights, axis=0),
                    tf.expand_dims(input_tensor, axis=1)
                ), name="Distance")  # [batch_size, neurons]

                # Calculate the Gaussian of the activity. Units with distances
                # closer to 0 will have activities closer to 1.
                activity = tf.exp(
                    tf.multiply(tf.pow(dist, 2), c), name="Gaussian"
                )

                # Convert the activity into a softmax probability distribution
                if self._softmax_activity:
                    activity = tf.divide(
                        tf.exp(activity),
                        tf.expand_dims(
                            tf.reduce_sum(tf.exp(activity), axis=1),
                            axis=-1
                        ),
                        name="Softmax"
                    )

                return tf.identity(activity, name="Output")

    def train(self, data, step_offset=0):
        """ Train the network on the data provided by the input tensor.
        :param num_inputs: The total number of inputs in the data-set. Used to
                            determine batches per epoch
        :param writer: The summary writer to add summaries to. This is created
                        by the caller so when we stack layers we don't end up
                        with duplicate outputs. If `None` then no summaries
                        will be written.
        :param step_offset: The offset for the global step variable so I don't
                            accidentally overwrite my summaries
        """

        # Build the TensorFlow dataset pipeline per the standard tutorial.
        if self._trained:
            LOGGER.warning("Model is already trained.")

        # initialize the input fitting dataset
        # the fitting input tensor is directly integrated into the
        # graph, which is why graph creation is postponed until fitting,
        # when we know the actual data of our input
        with self._graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices(
                data.astype(np.float32)
            )
            # number of samples and number of dimensions in our data
            num_inputs, self._dim = data.shape

            if self._initialization_method == "sample":
                samples = data.values[np.random.choice(
                    data.shape[0], self._m * self._n, replace=False
                ), :]
                self._init_samples = tf.convert_to_tensor(
                    samples, dtype=tf.float32
                )

            dataset = dataset.repeat()
            dataset = dataset.batch(self._batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            self._input_tensor = next_element

            # Create the ops and put them on the graph
            self._initialize_tf_graph()

            init_op = tf.global_variables_initializer()
            self._sess.run([init_op])

        # Divide by num_gpus to avoid accidentally training on the same data a
        # bunch of times
        batches_per_epoch = (
            num_inputs // self._batch_size // max(len(self._gpus), 1)
        )

        total_batches = batches_per_epoch * self._max_epochs
        # Get how many batches constitute roughly 10 percent of the total for
        # recording summaries
        global_step = step_offset

        LOGGER.info("Training self-organizing Map")
        for epoch in range(self._max_epochs):
            LOGGER.info("Epoch: %d/%d", epoch, self._max_epochs)
            for batch in range(batches_per_epoch):
                current_batch = batch + (batches_per_epoch * epoch)
                global_step = current_batch + step_offset
                percent_complete = current_batch / total_batches
                LOGGER.debug(
                    "\tBatch %d/%d - %.2f%% complete",
                    batch,
                    batches_per_epoch,
                    percent_complete * 100
                )
                # Only do summaries when a SummaryWriter has been provided
                self._sess.run(
                    self._training_op,
                    feed_dict={self._epoch: epoch}
                )

        self._trained = True
        return global_step

    @property
    def output_weights(self):
        """
        :return: The weights of the trained SOM as a NumPy array, or `None`
                    if the SOM hasn't been trained
        """
        if self._trained:
            return np.array(self._sess.run(self._weights))

        return None

    def map_to_nodes(self, data):
        """Map data to the closest node in the map."""
        # initialize dataset
        self._sess.run(
            self._prediction_input.initializer, feed_dict={self._invar: data}
        )

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
        self._sess.run(
            self._prediction_input.initializer, feed_dict={self._invar: data}
        )
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


class SelfOrganizingMap(BaseEstimator, TransformerMixin):
    """SOM abstraction for usage as a scikit learn transformer."""

    def __init__(
            self,
            m, n,
            max_epochs=10,
            checkpoint_dir="checkpoints",
            initialization_method="sample",
            restore_path=None,
    ):
        """Expose a subset of tested parameters for external tuning."""
        self._model = TFSom(m=m, n=n, max_epochs=max_epochs)

    @classmethod
    def load(cls, path):
        """Load configuration and state from saved configuration."""
        pass

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

    def __init__(self, m=10, n=10, batch_size=1024):
        self._model = TFSom(m, n, batch_size=batch_size)
        self.history = []

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, *_):
        self._model.train(X)
        weights = pd.DataFrame(
            self._model.output_weights, columns=X.columns
        )
        self.history.append({
            "data": weights,
            "mod": weights.index,
        })
        return weights
