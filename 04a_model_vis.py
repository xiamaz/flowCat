"""Example for the generation of saliency plots"""
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import vis.utils as vu
import tensorflow.keras.utils as ku
from vis.visualization import visualize_saliency

from flowcat import utils, io_functions, som_dataset


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

    def channel_occlusion(self, cases: list, som_sequence) -> list:
        tube_markers = [(t, m, i) for t, mm in self.config["tubes"].items() for i, m in enumerate(mm["channels"])]

        entropies = defaultdict(list)
        len_cases = len(cases)
        for i, case in enumerate(cases):
            print(f"[{i}/{len_cases} {case}")
            for tube, marker, index in tube_markers:
                xdata, ydata = som_sequence.get_batch_by_label([case.id])
                xdata[int(tube) - 1][:, :, :, index] = 0
                pred = self.model.predict(xdata)
                entropy = - np.sum(ydata[0] * np.log(pred))
                entropies[(tube, marker)].append(entropy)

        avg_results = []
        for (tube, marker), values in entropies.items():
            avg_results.append((tube, marker, np.mean(values), np.std(values)))
        return avg_results
        # print(sorted(avg_results, key=lambda t: t[2], reverse=True))


def main(data: utils.URLPath, meta: utils.URLPath, reference: utils.URLPath, model: utils.URLPath):
    data, meta, soms, model = map(utils.URLPath, [
        "/data/flowcat-data/mll-flowdata/decCLL-9F",
        "output/0-final-dataset/train.json.gz",
        "output/som-fix-test/soms-test/som_r4_1",
        "output/0-final/classifier-minmax-new",
    ])
    dataset = io_functions.load_case_collection(data, meta)
    soms = som_dataset.SOMDataset.from_path(soms)
    model = SaliencySOMClassifier.load(model)
    val_dataset = model.get_validation_data(dataset)
    val_seq = model.create_sequence(soms)

    # printing out weights and biases, unsure whether they actually contain
    # information
    # in theory we could extend that to attempt to describe them as gates
    tube = "3"
    weights, biases = model.model.layers[int(tube) + 2].get_weights()
    for j, chname in enumerate(model.config["tubes"][tube]["channels"]):
        ch_mean_weight = np.mean(weights[:, :, j, :])
        print(j, chname, ch_mean_weight)

    for i in range(weights.shape[-1]):
        mean_weight = np.mean(weights[:, :, :, i])
        print(i, mean_weight, biases[i])
        for j, chname in enumerate(model.config["tubes"]["1"]["channels"]):
            print(i, j, chname)
            print(weights[:, :, j, i])

    # zero out specific columns and see how that impacts performance
    output = utils.URLPath("output/0-final/model-analysis/occlusion")
    for group in model.config["groups"]:
        print(group)
        sel_cases = val_dataset.filter(groups=[group])
        avg_results = model.channel_occlusion(sel_cases, val_seq)
        print(sorted(avg_results, key=lambda t: t[2], reverse=True))
        io_functions.save_json(avg_results, output / f"{group}_avg_std.json")

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

    for tube in model.config["tubes"]:
        print("Tube", tube)
        print("\n".join(": ".join((t[0], str(t[1]))) for t in sorted(max_markers[tube], key=lambda t: t[1], reverse=True)))


    c_model = MLLDATA / "mll-sommaps/models/relunet_samplescaled_sommap_6class/model_0.h5"
    c_labels = MLLDATA / "mll-sommaps/output/relunet_samplescaled_sommap_6class/test_labels.json"
    c_preds = MLLDATA / "mll-sommaps/models/relunet_samplescaled_sommap_6class/predictions_0.csv"
    c_config = MLLDATA / "mll-sommaps/output/relunet_samplescaled_sommap_6class/config.json"
    c_cases = MLLDATA / "mll-flowdata/CLL-9F"
    c_sommaps = MLLDATA / "mll-sommaps/sample_maps/selected1_toroid_s32"
    c_misclass = MLLDATA / "mll-sommaps/misclassifications/"
    c_tube = [1, 2]

    # load datasets
    somdataset = sd.SOMDataset.from_path(c_sommaps)
    cases = cc.CaseCollection.from_path(c_cases, how="case_info.json")

    # filter datasets
    test_labels = flowutils.load_json(c_labels)

    filtered_cases = cases.filter(labels=test_labels)
    somdataset.data[1] = somdataset.data[1].loc[test_labels, :]

    # get mapping
    config = flowutils.load_json(c_config)
    groupinfo = mappings.GROUP_MAPS[config["c_groupmap"]]

    dataset = cd.CombinedDataset(
        filtered_cases, {dd.Dataset.from_str('SOM'): somdataset, dd.Dataset.from_str('FCS'): filtered_cases}, group_names = groupinfo['groups'])

    # modify mapping
    dataset.set_mapping(groupinfo)

    xoutputs = [loaders.loader_builder(
        loaders.Map2DLoader.create_inferred, tube=1,
        sel_count="counts",
        pad_width=1,
    ),
        loaders.loader_builder(
        loaders.Map2DLoader.create_inferred, tube=2,
        sel_count="counts",
        pad_width=1,
    )]

    dataset = loaders.DatasetSequence.from_data(
        dataset, xoutputs, batch_size=1, draw_method="sequential")

    predictions = pd.read_csv(c_preds, index_col=0)

    predictions = add_correct_magnitude(predictions)
    predictions = add_infiltration(predictions, cases)

    misclass_labels = ['507777582649cbed8dfb3fe552a6f34f8b6c28e3']

    for label in misclass_labels:
        label_path = pathlib.Path(f"{c_misclass}/{label}")
        if not label_path.exists():
            label_path.mkdir()

        case = cases.get_label(label)

        #get the actual and the predicited class
        corr_group = predictions.loc[case.id, "correct"]
        pred_group = predictions.loc[case.id, "pred"]
        classes = [corr_group,pred_group]

        gradients = plotting.calc_saliency(
            dataset, case, c_model, classes = classes)

        for tube in c_tube:

            heatmaps = plotting.draw_saliency_heatmap(case, gradients, classes, tube)
            for idx,heatmap in enumerate(heatmaps):
                plotting.save_figure(heatmap,f"{c_misclass}/{label}/{classes[idx]}_tube_{tube}_saliency_heatmap.png")

            scatterplots = plotting.plot_tube(case, tube, gradients[tube - 1], classes=classes, sommappath=c_sommaps)
            for idx,scatterplot in enumerate(scatterplots):
                plotting.save_figure(scatterplot,f"{c_misclass}/{label}/{classes[idx]}_tube_{tube}_scatterplots.png")


if __name__ == "__main__":
    main()
