"""
Create simple plots for SOMs and other objects.
"""
import logging
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)


def scale_weights_to_colors(weights):
    """Scale all given channels similar to tf.summary.image:
    if all positive:
        rescale to largest 1.0
    else:
        TODO: not implemented yet
        shift 0.0 to 0.5 and rescale to either min at 0 or max at 255
    """
    minimum = np.min(weights)
    if minimum < 0:
        LOGGER.warning(
            "Negative weights will not be transformed identically to tensorflow"
        )
    scaled = (weights - minimum) / (np.max(weights) - minimum)
    return scaled


def plot_color_grid(griddata, scale=True):
    """Plot array with 3 color channels as RGB values."""
    if scale:
        griddata = scale_weights_to_colors(griddata)

    fig = Figure()
    axes = fig.add_subplot(111)
    axes.imshow(griddata, interpolation="nearest")
    axes.set_axis_off()

    # Saving the figure
    FigureCanvas(fig)
    fig.tight_layout()
    return fig


def plot_color_grid(griddata, scale=True):
    """Plot array with 3 color channels as RGB values."""
    if scale:
        griddata = scale_weights_to_colors(griddata)

    fig = Figure()
    axes = fig.add_subplot(111)
    axes.imshow(griddata, interpolation="nearest")
    axes.set_axis_off()

    # Saving the figure
    FigureCanvas(fig)
    fig.tight_layout()
    return fig


def plot_som_grid(somdata, channels=None):
    """Plot a single som object with same visuals as tf.summary.image.
    """
    imgdata = somdata.data
    if channels is None:
        channels = channels.columns[:3]

    arr = np.stack([
        imgdata[c].values if c is not None else np.zeros(imgdata.shape[0])
        for c in channels
    ], axis=1)
    return plot_color_grid(arr, scale=True)
