"""Plot results from retraining for comparison purposes."""
import pandas as pd
import matplotlib.pyplot as plt
from flowcat import utils, io_functions, mappings


def load_datasets(data_path):
    datasets = {}
    for d in filter(lambda d: d.is_dir(), data_path.iterdir()):
        datasets[d.name] = {
            "data": io_functions.load_case_collection(d, d + ".json"),
            "config": io_functions.load_json(d + "_config.json"),
        }
    return datasets


def merged_data(datasets, group, tube):
    joined_data = {}
    for name in datasets:
        gdata = datasets[name]["data"].filter(groups=[group])
        samples = [c.get_tube(tube, kind="som").get_data().get_dataframe() for c in gdata]
        joined = pd.concat(samples)
        joined_data[name] = joined
    return joined_data


def plot_hexplot(data, channels, title, ax):
    channel_x, channel_y = channels
    ax.hexbin(data[channel_x], data[channel_y], cmap="Blues", gridsize=32)
    ax.set_ylabel(channel_y)
    ax.set_xlabel(channel_x)
    ax.set_title(title)
    return ax


def plot_hexplot_datasets(joined_datasets, channels, plotpath):
    print(len(joined_datasets))
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for (name, data), ax in zip(joined_datasets.items(), axes.flat):
        plot_hexplot(data, channels=channels, title=name, ax=ax)
    for ax in axes.flat:
        ax.label_outer()
    fig.tight_layout()
    plt.savefig(str(plotpath))
    plt.close("all")


def main():
    output = utils.URLPath("output/4-flowsom-cmp/retrain_figures")
    output.mkdir()
    data = utils.URLPath("output/4-flowsom-cmp/retrain_tests_32_learning_rate")
    # data = utils.URLPath("output/4-flowsom-cmp/retrain_tests_32_radius")

    datasets = load_datasets(data)
    groups = mappings.GROUPS
    tube = "1"
    group = "CLL"

    joined_datasets = merged_data(datasets, group, tube)
    # plot_hexplot_datasets(joined_datasets, ("CD45-KrOr", "SS INT LIN"), output / "radius_cll_cd_45_ss.png")
    plot_hexplot_datasets(joined_datasets, ("CD20-PC7", "CD5-PacBlue"), output / "learn_rate_cd20_cd5.png")
