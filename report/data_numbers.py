#!/usr/bin/env python3
# flake8: noqa
# pylint: skip-file
#
# Small script to output usable numbers of cases after filtering for tubes and
# marker identification
#
# Will also return marker composition
import os
import sys
# adding clustering library into pythonpath
sys.path.insert(0, "../clustering")

from argparse import ArgumentParser

from enum import IntEnum

from collections import defaultdict

import numpy as np
import pandas as pd

from matplotlib import cm, dates
from matplotlib.patches import Patch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from scipy.stats import gaussian_kde

from clustering.collection import CaseCollection
from report.plotting import plot_figure

DESC = """Overview numbers for the given case collection in the specified
bucket.

Numbers are inferred from the provided case_info json files.
"""


def conv_tubes(raw):
    """Convert comma-separated list of numbers to a proper python list of ints.
    """
    return [int(r) for r in raw.split(",")]


parser = ArgumentParser(description=DESC)
parser.add_argument(
    "path", help="Path to the directory containing files.",
    default="s3://mll-flowdata",
)
parser.add_argument(
    "--tubes", help="Specify used tubes.",
    default="1,2",
    type=conv_tubes
)
parser.add_argument(
    "--plotting", help="Enable plot generation.",
    action="store_true"
)
parser.add_argument(
    "--plotdir", help="Plotting directory",
    default="output/cohort_numbers"
)

args = parser.parse_args()

collection = CaseCollection(args.path, args.tubes)

# simulate our clustering environment to get realistic numbers
all_view = collection.create_view()

print("Total: {}\n".format(len(all_view)))

print("Cohorts")
print("\n".join(
    ["{}: {}".format(k, len(v)) for k, v in all_view.groups.items()]
))

print("\n")

print("Materials")
for t in args.tubes:
    tube_view = all_view.get_tube(t)
    print("Tube {}".format(t))
    print("\n".join([
        "{}: {}".format(k, len(v))
        for k, v in tube_view.materials.items()
    ]))
    print("Total: {}".format(len(tube_view)))
    print("\n")

    print("Channels: {}".format(len(tube_view.markers)))
    print("\n".join(tube_view.markers))
    print("\n")

dissimilar_tubes = []
for single in all_view:
    if len(set(
            [single.get_tube(p).material for p in args.tubes]
    )) != 1:
        dissimilar_tubes.append(single)

print("Dissimilar: {}".format(len(dissimilar_tubes)))
print("\n".join([
    "{}|{}: {}".format(
        d.group, d.id, ", ".join([
            str(d.get_tube(p).material) for p in args.tubes
        ])
    ) for d in dissimilar_tubes
]))


class Cohort(IntEnum):
    IN = 1
    CON = 2
    OUT = 3

    @classmethod
    def from_cohort_name(cls, name):
        if name == "normal":
            return cls.CON
        elif name in ["AML", "MM"]:
            return cls.OUT
        else:
            return cls.IN

    @classmethod
    def to_string(cls, enum):
        if enum == cls.IN:
            return "B-Cell Lymphoma"
        elif enum == cls.CON:
            return "normal"
        else:
            return "Other disorders"


def plot_cohorts(all_view, title="Cohort overview", path=""):
    """Plot cohort numbers in the given cohort, with optional
    coloring based on external groupings, such as ingroup, outgroup, control.
    """
    group_nums = {k: len(v) for k, v in all_view.groups.items()}

    sorted_keys = sorted(list(group_nums.keys()), key=Cohort.from_cohort_name)

    colors = [
        cm.Set1(Cohort.from_cohort_name(group))
        for group in sorted_keys
    ]
    plotpath = os.path.join(path, "cohorts_num")
    with plot_figure(plotpath, figsize=(10, 10)) as axes:
        bars = axes.bar(
            x=sorted_keys,
            height=[group_nums[v] for v in sorted_keys],
            color=colors,
        )
        for sbar in bars:
            height = sbar.get_height()
            axes.text(
                sbar.get_x() + sbar.get_width() / 2, height, str(height),
                ha="center", color="black", fontsize=12
            )

        patches = [
            Patch(facecolor=cm.Set1(n), label=Cohort.to_string(n))
            for n in Cohort
        ]
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)
        axes.legend(handles=patches)
        axes.set_title(title)
        axes.set_ylabel("Case number")


def plot_event_count(view, title="Event count plots", path=""):
    """Distribution of event counts."""
    all_event_counts = defaultdict(list)
    for cpath in view.get_tube(1).data:
        all_event_counts[cpath.parent.group].append(cpath.event_count)

    count_nums = []
    for name, data in all_event_counts.items():
        print("Counts {}".format(name))
        ccount = {}
        for percentile in [20000, 30000, 40000, 50000]:
            pdata = [d for d in data if d <= percentile]
            print(
                "<= {}: {} cases {} max".format(
                    percentile, len(pdata), max(pdata) if pdata else 0
                )
            )

            ccount["$\\leq$ {}".format(percentile)] = "{} (max {})".format(
                len(pdata), max(pdata) if pdata else 0
            )
        count_nums.append(ccount)

    count_df = pd.DataFrame(
        count_nums, index=all_event_counts.keys()
    )
    tablepath = os.path.join(path, "event_counts.latex")
    count_df.to_latex(tablepath, escape=False)

    plotpath = os.path.join(path, "event_distribution")
    with plot_figure(plotpath) as axes:
        # event_histo = np.histogram(all_event_counts, bins=bins)
        axes.boxplot(
            all_event_counts.values(),
            labels=all_event_counts.keys(),
        )
        axes.set_title(title)


def plot_date(view, title="Date distribution in each cohort.", path=""):
    """Plot the date distribution in each cohort as a
    density plot."""

    group_dates = defaultdict(list)
    for case in view.data:
        group_dates[case.group].append(dates.date2num(case.date))

    plotpath = os.path.join(path, "time_distribution")
    with plot_figure(plotpath, figsize=(20, 10)) as axes:
        for group, data in group_dates.items():
            axes.hist(data, bins=1000, label=group)
            axes.xaxis.set_major_locator(dates.YearLocator())
            axes.xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%Y"))
        axes.legend()


if args.plotting:
    os.makedirs(args.plotdir, exist_ok=True)
    plot_cohorts(all_view, "Cohort sizes CLL-9F", args.plotdir)
    plot_event_count(all_view, "Event count plots CLL-9F", args.plotdir)
    plot_date(all_view, "Date distribution CLL-9F", args.plotdir)
