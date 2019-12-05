import numpy as np


def plot_embedding(ax, data, labels, group_colors, title=""):
    """Plot the given 2D representation of the embedding information.

    Args:
        ax: Axis object for plotting.
        data: Numpy array of transformed input data with two columns.
        labels: List of groups for each sample.
        group_colors: Dict mapping groups to colors.
        title: Optional title for figure.
    """
    for i, (group, color) in enumerate(group_colors.items()):
        sel_dots = data[np.array([i == group for i in labels]), :]
        ax.scatter(sel_dots[:, 0], sel_dots[:, 1], label=group, s=16, marker="o", c=[color])
    # ax.legend()
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if title:
        ax.set_title(title)
    return ax
