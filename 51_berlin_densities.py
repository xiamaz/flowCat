"""
Compare channel intensities in berlin and munich dataset.
"""
from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from argmagic import argmagic

from flowcat import utils, io_functions


BERLIN_DATA = {
    "data_path": utils.URLPath("/data/flowcat-data/lb-data2"),
    "meta_path": utils.URLPath("output/50-berlin-data/dataset/known_groups.json")
}

MUNICH_DATA = {
    "data_path": utils.URLPath("/data/flowcat-data/mll-flowdata/decCLL-9F"),
    "meta_path": utils.URLPath("output/test-2019-08/dataset/train.json")
}


BERLIN_REMAP = {
    "CD45 KrOr": "CD45-KrOr",
}


def load_datasets():
    berlin_dataset = io_functions.load_case_collection(**BERLIN_DATA)
    munich_dataset = io_functions.load_case_collection(**MUNICH_DATA)
    return berlin_dataset, munich_dataset


def get_samples(cases, tube):
    return [case.get_tube(tube) for case in cases]


def data_to_channel(cases, tube, channel):
    fcs_samples = [case.get_tube(tube) for case in cases]
    fcs_data = [sample.get_data() for sample in fcs_samples]
    channel_data = [d.data[channel] for d in fcs_data]
    return channel_data


def data_to_channels(cases, tube, channels):
    fcs_samples = [case.get_tube(tube) for case in cases]
    fcs_data = [sample.get_data() for sample in fcs_samples]

    for fcsdata in fcs_data:
        fcsdata.rename(BERLIN_REMAP)

    channel_x, channel_y = channels
    channel_x_data = [d.data[channel_x] for d in fcs_data]
    channel_y_data = [d.data[channel_y] for d in fcs_data]
    return channel_x_data, channel_y_data


def plot_channel_densities(
        tube: str,
        channels: List[str],
        output: utils.URLPath
):
    """Plot the channel densities for a given dataset.

    Args:
        tube: Tube for which intensities should be generated.
        channels: List of channels used in generation.
        output: Output directory of plots.
    """
    berlin_dataset, munich_dataset = load_datasets()

    # berlin_sample = berlin_dataset.sample(10)
    # groups = list(berlin_sample.group_count.keys())

    # munich_sample = munich_dataset.sample(10, groups=groups)

    output = utils.URLPath("output/50-berlin_dataset/plot_channel_densities")

    # io_functions.save_json(berlin_sample.labels, output / "berlin_sample_labels.json")
    # io_functions.save_json(munich_sample.labels, output / "munich_sample_labels.json")

    berlin_sample_ids = io_functions.load_json(output / "berlin_sample_labels.json")
    munich_sample_ids = io_functions.load_json(output / "munich_sample_labels.json")
    berlin_sample = berlin_dataset.filter(labels=berlin_sample_ids)
    munich_sample = munich_dataset.filter(labels=munich_sample_ids)

    berlin_markers = berlin_sample.selected_markers

    from collections import defaultdict
    # find best match for each munich tube
    for tube, markers in munich_sample.selected_markers.items():
        counts = defaultdict(int)
        for btube, bmarkers in berlin_markers.items():
            bmarkers_name_only = [m.split()[0] for m in bmarkers]
            for marker in markers:
                marker = marker.replace("-", " ")
                mname = marker.split()[0]
                if marker in bmarkers:
                    print(btube, marker)
                    counts[btube] += 1
                elif mname in bmarkers_name_only:
                    print(btube, mname)
        print(counts)


    tube = "1"
    channels = ("CD45-KrOr", "SS INT LIN")

    output = utils.URLPath("output/50-berlin_dataset/plot_channel_densities/plots")
    output.mkdir()

    sns.set_style("white")

    # create hex bin plot
    for name, dataset in (("berlin", berlin_sample), ("munich", munich_sample)):
        for group in ("normal", "MCL", "CLL", "FL", "LPL", "MZL", "HCL"):
            datas_x, datas_y = data_to_channels(dataset, tube, channels)
            data_x = pd.concat(datas_x).reset_index(drop=True, inplace=False)
            data_y = pd.concat(datas_y).reset_index(drop=True, inplace=False)

            plt.figure()
            sns.jointplot(data_x, data_y, kind="hex")
            plt.savefig(str(output / f"hex_{name}_{group}.png"))
            plt.close("all")

    # create kde plot in one dimension
    group = "CLL"
    channel = "CD19 ECD"
    btubes = ("2", "3", "4")
    berlin_gsample = [c for c in berlin_sample if c.group == group]
    berlin_ts = [(tube, pd.concat(data_to_channel(berlin_gsample, tube, channel)).reset_index(drop=True)) for tube in btubes]

    mchannel = "CD19-APCA750"
    mtubes = ("1", "2", "3")
    munich_gsample = [c for c in munich_sample if c.group == group]
    munich_ts = [(tube, pd.concat(data_to_channel(munich_gsample, tube, mchannel)).reset_index(drop=True)) for tube in mtubes]

    fig, ax = plt.subplots()
    for tube, berlin_t in berlin_ts:
        sns.kdeplot(berlin_t, ax=ax, color="blue", label=f"Berlin {tube}")
    for tube, munich_t in munich_ts:
        sns.kdeplot(munich_t, ax=ax, color="red", label=f"Munich {tube}")
    fig.suptitle(f"{group} {channel} {mchannel}")
    fig.savefig(str(output / f"kde_munich_berlin_{group}_CD19.png"))

    # create kde plot after rescaling
    group = "CLL"
    channel = "Kappa FITC"
    btubes = ("2")
    berlin_gsample = [c for c in berlin_sample if c.group == group]
    berlin_ts = []
    for tube in btubes:
        datas = data_to_channel(berlin_gsample, tube, channel)
        transformed = []
        for data in datas:
            data = data.values
            data = data.reshape(-1, 1).astype("float32")
            tf = preprocessing.StandardScaler().fit_transform(data)
            transformed.append(tf.flatten())
        merged = np.concatenate(transformed)
        berlin_ts.append((tube, merged))

    mchannel = "Kappa-FITC"
    mtubes = ("2")
    munich_gsample = [c for c in munich_sample if c.group == group]
    munich_ts = []
    for tube in mtubes:
        datas = data_to_channel(munich_gsample, tube, mchannel)
        transformed = []
        for data in datas:
            data = data.values
            data = data.reshape(-1, 1).astype("float32")
            tf = preprocessing.StandardScaler().fit_transform(data)
            transformed.append(tf.flatten())
        merged = np.concatenate(transformed)
        munich_ts.append((tube, merged))

    fig, ax = plt.subplots()
    for tube, berlin_t in berlin_ts:
        sns.kdeplot(berlin_t, ax=ax, color="blue", label=f"Berlin {tube}")
    for tube, munich_t in munich_ts:
        sns.kdeplot(munich_t, ax=ax, color="red", label=f"Munich {tube}")
    fig.suptitle(f"{group} {channel} {mchannel}")
    fig.savefig(str(output / f"standard_kde_munich_berlin_{group}_kappa.png"))

    # for group in ("normal", "MCL", "CLL", "FL", "LPL", "MZL", "HCL"):
        # fig, ax = plt.subplots()
        # sns.kdeplot(

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.hexbin(berlin_ch_x_merged, berlin_ch_y_merged, cmap="Reds")
    # ax1.set_xlabel(channels[0])
    # ax1.set_ylabel(channels[1])
    # ax1.set_title("Berlin normals test")

    # ax2.hexbin(munich_ch_x_merged, munich_ch_y_merged, cmap="Blues")
    # ax2.set_xlabel(channels[0])
    # ax2.set_ylabel(channels[1])
    # ax2.set_title("Munich normals test")

    # fig.tight_layout()
    # fig.savefig("hexnormal.png")


if __name__ == "__main__":
    argmagic(plot_channel_densities)
