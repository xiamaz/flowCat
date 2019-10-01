# pylint: skip-file
# flake8: noqa
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from flowcat import io_functions, utils

data = utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F")
meta = utils.URLPath("output/0-munich-data/dataset/train.json")
meta_test = utils.URLPath("output/0-munich-data/dataset/test.json")

dataset = io_functions.load_case_collection(data, meta)
print(dataset)


def merge_cases(dataset, tube, channel):
    return np.concatenate(tuple(map(
        lambda d: d.data[:, d.channels.index(channel)],
        map(lambda c: c.get_tube(tube).get_data(), dataset)
    )))


# Create channel 1D distribution plots
groups = ["normal", "CLL", "HCL"]

channels = [
    ("1", "CD45-KrOr"),
    ("1", "CD19-APCA750"),
    ("1", "CD5-PacBlue"),
    ("2", "CD103-APC"),
]
sns.set_style("white")

tube, channel = channels[0]

for tube, channel in channels:
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1024)
    for group in groups:
        selected = dataset.filter(groups=[group]).sample(50)
        merged = np.concatenate(tuple(map(
            lambda d: d.data[:, d.channels.index(channel)],
            map(lambda c: c.get_tube(tube).get_data(), selected)
        )))
        sns.kdeplot(merged, ax=ax, label=group)

    ax.legend()
    ax.set_ylabel("Distribution as kernel density")
    ax.set_xlabel("Channel intensity")
    ax.set_title(f"{channel}")

    output = utils.URLPath("output/0-munich-data/figures-dataset")
    output.mkdir()
    plt.savefig(str(output / f"{channel}.png"))
    plt.close("all")

# Create over time distribution on a single group
import datetime
group = "normal"
selection = dataset.filter(groups=[group])
ranges = [(year, month) for year in range(2016, 2020) for month in range(1, 13)][:-11:4]
sns.set_style("white")
sns.set_palette(sns.cubehelix_palette(len(ranges)))

file_path = output / f"timeplot.png"

fig, ax = plt.subplots()
ax.set_xlim(0, 1024)
for i, end_month in enumerate(ranges[1:]):
    end = datetime.date(*end_month, 1) - datetime.timedelta(days=1)
    start = datetime.date(*ranges[i], 1)
    label = f"{start.strftime('%m/%y')}-{end.strftime('%m/%y')}"
    print(label)
    time_cases = selection.filter(date=(start, end))
    if len(time_cases) == 0:
        continue
    merged = merge_cases(time_cases, "1", "CD45-KrOr")
    sns.kdeplot(merged, ax=ax, label=label)

ax.legend()
ax.set_ylabel("Distribution as kernel density")
ax.set_xlabel("Channel intensity")
ax.set_title(f"Normal CD45-KrOr over time")
fig.tight_layout()
plt.savefig(str(file_path))
plt.close("all")
