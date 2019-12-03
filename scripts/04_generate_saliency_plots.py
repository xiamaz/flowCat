"""Example for the generation of saliency plots"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import vis.utils as vu
from vis.visualization import visualize_saliency

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from flowcat import utils, io_functions, som_dataset
from flowcat import mappings
from flowcat.plots import som as plt_som


def calculate_bmu_indexes():
    mapdata = tf.placeholder(tf.float32, shape=(None, None), name="som")
    fcsdata = tf.placeholder(tf.float32, shape=(None, None), name="fcs")
    squared_diffs = tf.pow(tf.subtract(
        tf.expand_dims(mapdata, axis=0),
        tf.expand_dims(fcsdata, axis=1)), 2)
    diffs = tf.reduce_sum(squared_diffs, 2)
    bmu = tf.argmin(diffs, axis=1)
    return bmu


class SaliencySOMClassifier:
    layer_idx = -1

    def __init__(self, model, binarizer, config, data_ids: dict = None):
        self.model = model
        self.config = config
        self.binarizer = binarizer
        self.data_ids = data_ids

    @classmethod
    def load(cls, path: utils.URLPath):
        """Load classifier model from the given path."""
        config = io_functions.load_json(path / "config.json")

        model = keras.models.load_model(
            str(path / "model.h5"),
        )
        model.layers[-1].activation = keras.activations.linear
        model = vu.utils.apply_modifications(model)

        binarizer = io_functions.load_joblib(path / "binarizer.joblib")

        data_ids = {
            "validation": io_functions.load_json(path / "ids_validate.json"),
            "train": io_functions.load_json(path / "ids_train.json"),
        }
        return cls(model, binarizer, config, data_ids=data_ids)

    def get_validation_data(self, dataset: som_dataset.SOMDataset) -> som_dataset.SOMDataset:
        return dataset.filter(labels=self.data_ids["validation"])

    def create_sequence(
        self,
        dataset: som_dataset.SOMDataset,
        batch_size: int = 128
    ) -> som_dataset.SOMSequence:

        if isinstance(dataset, som_dataset.SOMDataset):
            def getter(data, tube):
                return data.get_tube(tube, kind="som").data
        else:
            def getter(data, tube):
                return data.get_tube(tube, kind="som").get_data().data

        seq = som_dataset.SOMSequence(
            dataset, self.binarizer,
            get_array_fun=getter,
            tube=self.config["tubes"],
            batch_size=batch_size,
            pad_width=self.config["pad_width"],
        )
        return seq

    def calculate_saliency(self, som_sequence, case, group, maximization=False):
        """Calculates the saliency values / gradients for the case, model and
        each of the classes.
        Args:
            dataset: SOMMapDataset object.
            case: Case object for which the saliency values will be computed.
            group: Select group.
            layer_idx: Index of the layer for which the saleincy values will be
                computed.
            maximization: If true, the maximum of the saliency values over all
                channels will be returned.
        Returns:
            List of gradient values sorted first by tube and then class (e.g.
                [[tube1_class1,tube1_class1][tube2_class1,tube2_class2]]).
        """
        xdata, _ = som_sequence.get_batch_by_label(case.id)
        input_indices = [*range(len(xdata))]
        gradients = visualize_saliency(
            self.model,
            self.layer_idx,
            self.config["groups"].index(group),
            seed_input=xdata,
            input_indices=input_indices,
            maximization=maximization
        )
        return gradients


def get_channel_data(model, array, tube, channel=None) -> "array":
    channel_info = model.config["tubes"]
    tube_index = list(channel_info.keys()).index(tube)
    if channel:
        channel_index = channel_info[tube]["channels"].index(channel)
        return array[tube_index][..., channel_index]
    return array[tube_index]


def plot_saliency_som_map(model, somdata, gradient, tube, channels):
    """Plot saliency and color channels as plots."""
    rgb = np.stack([
        get_channel_data(model, somdata, tube, channel)
        for channel in channels
    ], axis=-1)
    grad_max = np.max(get_channel_data(model, gradient, tube), axis=-1)

    fig = Figure()
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.1])
    ax_main = fig.add_subplot(grid[:2, :2])
    ax_main.imshow(plt_som.scale_weights_to_colors(rgb), interpolation="nearest")
    ax_main.set_axis_off()
    ax_main.imshow(grad_max, interpolation="nearest", alpha=0.60, cmap="gray")
    ax_main.set_title("Combined")
    patches = [
        mpl.patches.Patch(color=color, label=channel)
        for channel, color in zip(channels, ("red", "green", "blue"))
    ]
    ax_rgb = fig.add_subplot(grid[0, 2])
    ax_rgb.imshow(plt_som.scale_weights_to_colors(rgb), interpolation="nearest")
    ax_rgb.set_axis_off()
    ax_rgb.set_title("Channels")
    ax_mask = fig.add_subplot(grid[1, 2])
    ax_mask.imshow(grad_max, interpolation="nearest", cmap="gray")
    ax_mask.set_axis_off()
    ax_mask.set_title("Saliency")
    ax_legend = fig.add_subplot(grid[2, :])
    ax_legend.legend(handles=patches, framealpha=0.0, ncol=3, loc="center")
    ax_legend.set_axis_off()
    FigureCanvas(fig)
    fig.tight_layout()
    return fig


def annotate_fcs_data(model, bmu_calc, session, case, tube, somdata, gradient):
    channels = model.config["tubes"][tube]["channels"]
    fcssample = case.get_tube(tube, kind="fcs")
    fcsdata = fcssample.get_data().align(channels)
    fcsdata = MinMaxScaler().fit_transform(fcsdata.data)
    tube_index = list(model.config["tubes"].keys()).index(tube)
    somdata = somdata[tube_index].reshape((-1, len(channels)))
    mapping = session.run(bmu_calc, feed_dict={"fcs:0": fcsdata, "som:0": somdata})
    gradient_values = gradient[tube_index].reshape((-1, len(channels)))
    fcs_gradient = gradient_values[mapping, :]
    return channels, fcsdata, fcs_gradient


def plot_saliency_scatterplot(model, bmu_calc, session, case, tube, xdata, gradients, norm=None, all_maximum=True):
    channels, fcs, grads = annotate_fcs_data(model, bmu_calc, session, case, tube, xdata, gradients)

    t1_view = mappings.PLOT_2D_VIEWS[tube]
    view_num = len(t1_view)
    cols = 4
    rows = int(np.ceil(view_num / cols))
    width = 4
    height = 4

    fig = Figure(figsize=(width * cols, height * rows))
    axes = fig.subplots(nrows=rows, ncols=cols)
    for ax, (chx, chy) in zip(axes.flatten(), t1_view):
        chx_index = channels.index(chx)
        chy_index = channels.index(chy)
        # ch_grads = np.max(grads[:, [chx_index, chy_index]], axis=-1)
        ch_grads = np.max(grads, axis=-1)
        ax.scatter(fcs[:, chx_index], fcs[:, chy_index], marker=".", c=ch_grads, s=1, cmap="Greys", norm=norm)
        ax.set_xlabel(chx)
        ax.set_ylabel(chy)

    for ax in axes.flatten()[view_num:]:
        ax.set_axis_off()
    FigureCanvas(fig)
    fig.tight_layout()
    return fig


def main(data: utils.URLPath, meta: utils.URLPath, reference: utils.URLPath, model: utils.URLPath):
    data, meta, soms, model = map(utils.URLPath, [
        "/data/flowcat-data/mll-flowdata/decCLL-9F",
        "output/0-final-dataset/train.json.gz",
        "output/som-fix-test/soms-test/som_r4_1",
        "output/0-final/classifier-minmax-new",
    ])
    sommodel = utils.URLPath("output/som-fix-test/unjoined-ref")
    sommodel = io_functions.load_casesom(sommodel)

    output = utils.URLPath("output/0-final/model-analysis/saliency")
    output.mkdir()
    dataset = io_functions.load_case_collection(data, meta)
    soms = som_dataset.SOMDataset.from_path(soms)
    model = SaliencySOMClassifier.load(model)
    val_dataset = model.get_validation_data(dataset)
    val_seq = model.create_sequence(soms)

    selected_labels = [
        "c3a6098bd5216c7d1f958396dd31bd6ef1646c18",
        "df726c162ed728c2886107e665ad931e5bf0baae",
        "3eb03bea6651c302ac013f187b288ee990889b29",
        "e539b3ec66b1c9d7a0aae1fbd37c19c7ac86a18c",
        "762a2a19d1913383f41ead7b5ef74a8133d67847",
        "bbfafb3d9053e212279aaada5faf23eddf4a5926",
        "9503bfad60524615a06613cfbffa3861fb66ede3",
    ]
    sel_dataset = dataset.filter(labels=selected_labels)

    # annotate each fcs point with saliency info
    session = tf.Session()
    bmu_calc = calculate_bmu_indexes()

    normalize = mpl.colors.Normalize(vmin=0, vmax=1)

    case = sel_dataset[0]
    for case in sel_dataset:
        case_output = output / f"{case.id}_g{case.group}"
        case_output.mkdir()
        print("Plotting", case)

        # plot som and saliency activations
        result = model.calculate_saliency(val_seq, case, case.group, maximization=False)

        xdata, _ = val_seq.get_batch_by_label([case.id])
        xdata = [x[0, ...] for x in xdata]

        for tube in ("1", "2", "3"):
            fig = plot_saliency_som_map(model, xdata, result, tube, ("CD45-KrOr", "SS INT LIN", "CD19-APCA750"))
            fig.savefig(str(case_output / f"t{tube}_overlay.png"))

            fig = plot_saliency_scatterplot(model, bmu_calc, session, case, tube, xdata, result, norm=normalize)
            fig.savefig(str(case_output / f"t{tube}_scatter_saliency.png"))

    for case in sel_dataset:
        case_output = output / f"maxall_{case.id}_g{case.group}"
        case_output.mkdir()
        print("Plotting", case)

        # plot som and saliency activations
        result = model.calculate_saliency(val_seq, case, case.group, maximization=False)
        for r in result:
            print("Max", np.max(r))

        xdata, _ = val_seq.get_batch_by_label([case.id])
        xdata = [x[0, ...] for x in xdata]

        for tube in ("1", "2", "3"):
            fig = plot_saliency_som_map(model, xdata, result, tube, ("CD45-KrOr", "SS INT LIN", "CD19-APCA750"))
            fig.savefig(str(case_output / f"t{tube}_overlay.png"))

            fig = plot_saliency_scatterplot(model, bmu_calc, session, case, tube, xdata, result, norm=normalize)
            fig.savefig(str(case_output / f"t{tube}_scatter_saliency.png"))

    # case_som = soms.get_labels([case.id]).iloc[0]
    hcls = val_dataset.filter(groups=["HCL"])
    from collections import defaultdict
    max_vals = defaultdict(lambda: defaultdict(list))
    mean_vals = defaultdict(lambda: defaultdict(list))
    for case in hcls:
        print(case)
        gradient = model.calculate_saliency(val_seq, case, case.group, maximization=False)
        for i, (tube, markers) in enumerate(model.config["tubes"].items()):
            tgrad = gradient[i]
            for i, marker in enumerate(markers["channels"]):
                mgrad = tgrad[:, :, i]
                gmax = np.max(mgrad)
                max_vals[tube][marker].append(gmax)
                gmean = np.mean(mgrad)
                mean_vals[tube][marker].append(gmean)
    max_markers = defaultdict(list)
    for tube, markers in model.config["tubes"].items():
        for marker in markers["channels"]:
            print("Max", tube, marker, np.mean(max_vals[tube][marker]))
            print("Mean", tube, marker, np.mean(mean_vals[tube][marker]))
            max_markers[tube].append((marker, np.mean(max_vals[tube][marker])))


if __name__ == "__main__":
    main()
