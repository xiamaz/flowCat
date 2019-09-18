"""
Create plots for confusion matrices.
"""
import logging
import itertools

import numpy as np
import pandas as pd

from scipy.cluster import hierarchy

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


LOGGER = logging.getLogger(__name__)


def normalize_confusion_matrix(confusion_matrix):
    """Normalize the given confusion matrix."""
    classes = confusion_matrix.columns
    confusion_matrix = confusion_matrix.values
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    return pd.DataFrame(confusion_matrix, index=classes, columns=classes)


def matrix_to_sizes(matrix, sizes):
    """Expand the confusion matrix to the given sizes by duplicating rows and
    columns.

    Args:
        matrix: Numpy array.
        sizes: List of sizes for each column in the original data.
    Returns:
        Expanded matrix.
    """
    # tile horizontally
    hreplicated = []
    for i, size in enumerate(sizes):
        hreplicated.append(np.tile(matrix[[i], :], (size, 1)))
    matrix = np.concatenate(hreplicated)

    # tile vertically
    vreplicated = []
    for i, size in enumerate(sizes):
        vreplicated.append(np.tile(matrix[:, [i]], (1, size)))
    matrix = np.concatenate(vreplicated, axis=1)
    return matrix


def draw_confusion_matrix(
        axes, confusion_matrix, sizes=None, cmap="Blues"
):
    """Draw a confusion matrix on the given axes.
    Args:
        axes:
        confusion_matrix:
        classes:
    Returns:
        Axes containing confusion matrix.
    """
    classes = list(confusion_matrix.columns)

    confusion_matrix = confusion_matrix.values

    # set tick mark values and confusion matrix values for image plot
    tick_marks = np.arange(len(classes), dtype=float)
    confusion_img = confusion_matrix.copy()
    if sizes is not None:
        for i, size in enumerate(sizes):
            if size > 1:
                # shift the tickmarks so that they are in the center of larger
                # merged cells
                tick_marks[i] = np.arange(tick_marks[i], size + tick_marks[i], dtype=float).mean()
                tick_marks[i + 1:] = tick_marks[i + 1:] + size - 1

        confusion_img = matrix_to_sizes(confusion_matrix, sizes)

    axes.imshow(confusion_img, interpolation='nearest', cmap=cmap)
    axes.set_xticks(tick_marks)
    axes.set_xticklabels(classes)
    axes.set_yticks(tick_marks)
    axes.set_yticklabels(classes)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')

    # print the number values in each cell
    if "float" in confusion_img.dtype.name:
        fmt = ".2f"
    else:
        fmt = "d"
    thresh = confusion_img.max() / 2.
    for i, j in itertools.product(range(len(tick_marks)),
                                  range(len(tick_marks))):
        axes.text(
            tick_marks[j], tick_marks[i], format(confusion_matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > thresh else "black")

    return axes


def draw_dendrogram(axes, confusion_matrix):
    """Plot a dendrogram based on hierarchical clustering of the input
    confusion matrix.
    Args:
        axes: Axes used for plotting.
        confusion_matrix: Pandas dataframe or numpy array.
        classes: Labels to be used for plotting. Required for numpy array.
            Optional for dataframes. Will be inferred from column names if not
            given for the latter.
    Returns:
        Plotted axes.
    """
    classes = list(confusion_matrix.columns)

    pairwise_dists = hierarchy.distance.pdist(
        confusion_matrix, metric='euclidean')
    dist_linkage = hierarchy.linkage(pairwise_dists, method='single')

    hierarchy.dendrogram(
        dist_linkage, show_contracted=True, labels=classes, ax=axes)
    return axes


def plot_confusion_matrix(
        confusion_matrix,
        normalize=False,
        title="Confusion matrix",
        cmap="Blues",  # pylint: disable=no-member
        filename="confusion.png",
        dendroname="dendro.png",
        sizes=None, colorbar=False
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    Args:
        confusion_matrix:
        classes:
        normalize:
        title:
        cmap:
        filename:
    """
    if normalize:
        LOGGER.debug("Normalized confusion matrix.")
        confusion_matrix = normalize_confusion_matrix(confusion_matrix)
    else:
        LOGGER.debug('Confusion matrix, without normalization')
        confusion_matrix = confusion_matrix.astype('int')

    if filename is not None:
        fig = Figure()
        axes = fig.add_subplot(111)
        axes.set_title(title)
        draw_confusion_matrix(axes, confusion_matrix, sizes=sizes)
        FigureCanvas(fig)
        fig.tight_layout()
        fig.savefig(str(filename), dpi=300)

    if dendroname is not None:
        fig = Figure()
        axes = fig.add_subplot(111)
        draw_dendrogram(axes, confusion_matrix)
        FigureCanvas(fig)
        fig.tight_layout()
        fig.savefig(str(dendroname), dpi=300)
