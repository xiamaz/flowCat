"""
Create simple plots for SOMs and other objects.
"""
import logging
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from . import som


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


def plot_som_grid(somdata, destpath, channels=None):
    """Plot a single som object with same visuals as tf.summary.image.
    """
    imgdata = somdata.data
    if channels is None:
        channels = channels.columns[:3]

    arr = np.stack([
        imgdata[c].values if c is not None else np.zeros(imgdata.shape[0])
        for c in channels
    ], axis=1)
    scaled = scale_weights_to_colors(arr)

    scaled = np.reshape(scaled, (32, 32, 3))

    fig = Figure()
    axes = fig.add_subplot(111)
    axes.imshow(scaled, interpolation="nearest")
    axes.set_axis_off()
    axes.set_title("/".join(map(str, channels)))

    # Saving the figure
    FigureCanvas(fig)
    fig.tight_layout()
    fig.savefig(str(destpath))
