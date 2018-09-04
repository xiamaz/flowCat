"""
3 dimensional visualizations of the consensus SOM based upsampling.
"""
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def histogram_3d(data, plotpath, ax=None):
    """3D visualization of the histogram distribution of individual events
    across the self organizing map."""
    fig = Figure()

    axes = fig.add_subplot(111, projection="3d")
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = np.zeros_like(x)

    dx = 1 * np.ones_like(x)  # width in x dimension
    dy = dx.copy()  # width in y dimension
    dz = data.iloc[:, 2]
    axes.bar3d(
        x, y, z,
        dx, dy, dz
    )
    axes.view_init(elev=45, azim=45)
    axes.set_title("Distribution of events in 3d SOM map")

    FigureCanvas(fig)
    fig.savefig(plotpath)


def heatmap_2d(data, plotpath, title="Heatmap"):
    """Draw a heatmap for the nodes using values in the given columns."""
    fig = Figure()

    axes = fig.add_subplot(111, projection="2d")
    axes.imshow(X=data,)
    axes.set_title(title)

    FigureCanvas(fig)
    fig.savefig(plotpath)


def countmap_2d(data, plotpath):
    """Map of the counts in the individual nodes."""
    counts = data["counts"]
    # approximate for square maps
    dims = np.round((np.sqrt(len(counts))))
    np.reshape(counts, (dims, dims))


def scatter_reference(data, plotpath, tube=1):
    """Plot reference scatter plots for the given data."""

    gatings = {
        1: [
            ["CD19-APCA750", "CD79b-PC5.5"],
            ["CD19-APCA750", "CD5-PacBlue"],
            ["CD20-PC7", "CD23-APC"],
            ["CD19-APCA750", "CD10-PE"],
            ["CD19-APCA750", "FMC7-FITC"],
            ["CD20-PC7", "CD5-PacBlue"],
            ["CD19-APCA750", "IgM-ECD"],
            ["CD10-PE", "FMC7-FITC"],
            ["SS INT LIN", "FS INT LIN"],
            ["CD45-KrOr", "SS INT LIN"],
            ["CD19-APCA750", "SS INT LIN"],
            ["CD19-APCA750", "CD10-PE"],
            ["CD19-APCA750", "FS INT LIN"],
        ],
        2: [
            ["CD19-APCA750", "Lambda-PE"],
            ["CD19-APCA750", "Kappa-FITC"],
            ["Lambda-PE", "Kappa-FITC"],
            ["CD19-APCA750", "CD22-PacBlue"],
            ["CD19-APCA750", "CD103-APC"],
            ["CD19-APCA750", "CD11c-PC7"],
            ["CD25-PC5.5", "CD11c-PC7"],
            ["Lambda-PE", "Kappa-FITC"],
            ["SS INT LIN", "FS INT LIN"],
            ["CD45-KrOr", "SS INT LIN"],
            ["CD19-APCA750", "SS INT LIN"],
        ],
        3: [
            ["CD3-ECD", "CD4-PE"],
            ["CD3-ECD", "CD8-FITC"],
            ["CD4-PE", "CD8-FITC"],
            ["CD56-APC", "CD3-ECD"],
            ["CD4-PE", "HLA-DR-PacBlue"],
            ["CD8-FITC", "HLA-DR-PacBlue"],
            ["CD19-APCA750", "CD3-ECD"],
            ["SS INT LIN", "FS INT LIN"],
            ["CD45-KrOr", "SS INT LIN"],
            ["CD3-ECD", "SS INT LIN"]
        ]
    }

    cur_gating = gatings[tube]
    num_horiz = int(np.ceil(np.sqrt(len(cur_gating))))
    num_vert = int(np.ceil(len(cur_gating) / num_horiz))

    print(len(cur_gating), num_horiz, num_vert)

    fig = Figure(figsize=(num_horiz * 5, num_vert * 5))

    for i, (xgate, ygate) in enumerate(cur_gating):
        axes = fig.add_subplot(num_vert, num_horiz, i + 1)

        x = data[xgate]
        y = data[ygate]

        axes.scatter(x, y, s=1, marker=".")
        axes.set_xlabel(xgate)
        axes.set_ylabel(ygate)
        axes.set_xlim(0, 1023)
        axes.set_ylim(0, 1023)

    fig.suptitle(f"Tube {tube}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    FigureCanvas(fig)
    fig.savefig(plotpath)


def visualize_weight_count(data):
    """Visualize three dimensional distribution of events across SOM map
    and show the network grid of the SOM, as well as some 2d plots of the SOM
    nodes."""
    pass


def get_count_coords(data, m=10, n=10):
    """Get counts from dataframe together with coordinates for plotting."""
    coords = pd.DataFrame(
        list(itertools.product(range(m), range(n)))
    )
    coords["counts"] = data["counts"]
    return coords


def main():
    output = Path("sommaps/plots")
    output.mkdir(parents=True, exist_ok=True)
    example_data = pd.read_csv(
        "sommaps/huge_s30_counts/00050ae5c8d7d4cd1d8aedb39e06edada7a8d7d4_t1.csv",
        index_col=0
    )
    example_raw = None

    count_data = get_count_coords(example_data, m=30, n=30)

    histogram_3d(count_data, output / "histogram_3d_counts")
    scatter_reference(example_data, output / "overview_scatter", tube=1)
