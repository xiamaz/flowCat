"""Basic information on FCS files."""
import sys
import pathlib
import argparse
import itertools

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import seaborn as sns
sns.set()

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent))

from flowcat.data import case
from flowcat.data import case_dataset


def get_case_sample(case, channel, tube=1):
    case.set_allowed_material([1, 2])
    t1 = case.get_tube(tube)
    datas = t1.data.data[channel].sample(100)
    return np.repeat(t1.date, 100), datas


def plot_expression_series(dataset, marker, tube, title, filename):
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for case in dataset:
        x, y = get_case_sample(case, marker, tube)
        ax.scatter(x, y, marker=".", s=1, c="black")

    ax.set_ylabel("Marker expression")
    ax.set_xlabel("Sample analysis date.")
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    ax.set_title(title)
    FigureCanvas(fig)
    fig.tight_layout()
    fig.savefig(filename)


def main():
    cases = case_dataset.CaseCollection.from_dir("/data/users/Max/MLL/data/newCLL-9F")

    def create_plot(marker, group, tube):
        test = cases.filter(groups=[group])
        title = f"Tube {tube}: {marker} over time in {group}"
        filename = f"{group}_{marker.replace(' ', '-')}_t{tube}.png"
        print("Generating", filename)
        plot_expression_series(test, marker, tube, title=title, filename=filename)

    tubes = [1]
    markers = ["SS INT LIN", "FS INT LIN", "CD45-KrOr", "CD19-APCA750"]
    # groups = ["HCL"]
    groups = set(cases.groups)
    list(itertools.starmap(create_plot, itertools.product(markers, groups, tubes)))


if __name__ == "__main__":
    main()
