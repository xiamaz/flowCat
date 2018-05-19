import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_som import SelfOrganizingMap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import logging
from scipy.spatial import distance_matrix
import fcsparser
from compile_cases import get_case_data, load_test

'''
An example usage of the TensorFlow SOM. Loads a data set, trains a SOM, and displays the u-matrix.
'''


def get_umatrix(weights, m, n):
    """ Generates an n x m u-matrix of the SOM's weights.

    Used to visualize higher-dimensional data. Shows the average distance between a SOM unit and its neighbors.
    When displayed, areas of a darker color separated by lighter colors correspond to clusters of units which
    encode similar information.
    :param weights: SOM weight matrix, `ndarray`
    :param m: Rows of neurons
    :param n: Columns of neurons
    :return: m x n u-matrix `ndarray`
    """
    umatrix = np.zeros((m * n, 1))
    # Get the location of the neurons on the map to figure out their neighbors. I know I already have this in the
    # SOM code but I put it here too to make it easier to follow.
    neuron_locs = list()
    for i in range(m):
        for j in range(n):
            neuron_locs.append(np.array([i, j]))
    # Get the map distance between each neuron (i.e. not the weight distance).
    neuron_distmat = distance_matrix(neuron_locs, neuron_locs)

    for i in range(m * n):
        # Get the indices of the units which neighbor i
        neighbor_idxs = neuron_distmat[i] <= 1  # Change this to `< 2` if you want to include diagonal neighbors
        # Get the weights of those units
        neighbor_weights = weights[neighbor_idxs]
        # Get the average distance between unit i and all of its neighbors
        # Expand dims to broadcast to each of the neighbors
        umatrix[i] = distance_matrix(np.expand_dims(weights[i], 0), neighbor_weights).mean()

    return umatrix.reshape((m, n))


def transform_non_linear(X):
    names = [n for n in X.columns if "LIN" not in n]
    X[names] = np.log1p(X[names])
    return X


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    graph = tf.Graph()
    with graph.as_default():
        # Make sure you allow_soft_placement, some ops have to be put on the CPU (e.g. summary operations)
        session = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        data, names, cases = get_case_data(sys.argv[1])
        num_inputs, dims = data.shape
        # log_transformer = FunctionTransformer(transform_non_linear)
        scaler = StandardScaler()
        trans_data = transform_non_linear(data)
        input_data = scaler.fit_transform(trans_data)
        # batch_size = 1024
        batch_size = 2048

        # Build the TensorFlow dataset pipeline per the standard tutorial.
        dataset = tf.data.Dataset.from_tensor_slices(input_data.astype(np.float32))
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        # This is more neurons than you need but it makes the visualization look nicer
        m = 10
        n = 10

        # Build the SOM object and place all of its ops on the graph
        som = SelfOrganizingMap(m=m, n=n, dim=dims, max_epochs=2, gpus=1, session=session, graph=graph,
                                input_tensor=next_element, batch_size=batch_size, initial_learning_rate=0.05)

        init_op = tf.global_variables_initializer()
        session.run([init_op])

        # Note that I don't pass a SummaryWriter because I don't really want to record summaries in this script
        # If you want Tensorboard support just make a new SummaryWriter and pass it to this method
        som.train(num_inputs=num_inputs)

        # weights = som.output_weights

        # umatrix = get_umatrix(weights, m, n)
        # fig = plt.figure()
        # plt.imshow(umatrix, origin='lower')
        # plt.show(block=True)

        testdata = load_test(cases, names)
        test = tf.convert_to_tensor(testdata)
        res = som.predict(test)
        print(res)
