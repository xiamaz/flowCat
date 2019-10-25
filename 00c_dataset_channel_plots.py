# pylint: skip-file
# flake8: noqa
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from flowcat import io_functions, utils, mappings

data = utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F")
meta = utils.URLPath("/data/flowcat-data/2019-10_paper_data/0-munich-data/dataset/train.json")
meta_test = utils.URLPath("/data/flowcat-data/2019-10_paper_data/0-munich-data/dataset/test.json")

output = utils.URLPath("/data/flowcat-data/2019-10_paper_data/00c-channel-intensities")
output.mkdir()

dataset = io_functions.load_case_collection(data, meta)
print(dataset)


def merge_cases(dataset, tube, channel):
    return np.concatenate(tuple(map(
        lambda d: d.data[:, d.channels.index(channel)],
        map(lambda c: c.get_tube(tube).get_data(), dataset)
    )))


# Create channel 1D distribution plots
groups = ["normal", "CLL", "HCL"]
colors = mappings.ALL_GROUP_COLORS

channels = [
    ("1", "CD45-KrOr"),
    ("1", "CD19-APCA750"),
    ("1", "CD5-PacBlue"),
    ("2", "CD103-APC"),
]
sns.set_style("white")

for tube, channel in channels:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1024)
    for group in groups:
        selected = dataset.filter(groups=[group]).sample(50)
        merged = np.concatenate(tuple(map(
            lambda d: d.data[:, d.channels.index(channel)],
            map(lambda c: c.get_tube(tube).get_data(), selected)
        )))
        sns.kdeplot(merged, ax=ax, label=group, color=colors[group])

    ax.legend()
    ax.set_ylabel("Density function")
    ax.set_xlabel("Channel intensity")
    ax.set_title(f"Kernel density {channel}")
    fig.tight_layout()
    plt.savefig(str(output / f"{channel}.png"), dpi=300)
    plt.close("all")

# Create over time distribution on a single group
import datetime
for group, shift in [("normal", 9), ("CLL", 9), ("HCL", 4)]:
    selection = dataset.filter(groups=[group])
    ranges = [(year, month) for year in range(2016, 2020) for month in range(1, 13)][:-11:4]


    base_color = colors[group]

    color_range = [i / 20.0 for i in range(-shift, len(ranges) - shift)]
    color_range = [[c + i for c in base_color] for i in color_range]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1024)
    for i, end_month in enumerate(ranges[1:]):
        end = datetime.date(*end_month, 1) - datetime.timedelta(days=1)
        start = datetime.date(*ranges[i], 1)
        label = f"{start.strftime('%m/%y')}-{end.strftime('%m/%y')}"
        time_cases = selection.filter(date=(start, end))
        print(label, time_cases)
        if len(time_cases) == 0:
            continue
        merged = merge_cases(time_cases, "1", "CD45-KrOr")
        sns.kdeplot(merged, ax=ax, label=label, color=color_range[i])

    ax.legend()
    ax.set_ylabel("Density function")
    ax.set_xlabel("Channel intensity")
    ax.set_title(f"{group} samples CD45-KrOr over time")
    fig.tight_layout()
    plt.savefig(str(output / f"{group}_timeplot.png"))
    plt.close("all")
