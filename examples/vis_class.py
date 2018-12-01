import os
import pathlib

import numpy as np
import pandas as pd

import pickle

from flowcat.data import case_dataset as cc
from flowcat.data import all_dataset as ad
from flowcat.data import loaders
from flowcat.visual import plotting


if "MLLDATA" in os.environ:
    MLLDATA = pathlib.Path(os.environ["MLLDATA"])
else:
    MLLDATA = pathlib.Path('../data')

def add_correct_magnitude(data):
    newdata = data.copy()
    valcols = [c for c in data.columns if c != "correct"]
    selval = np.vectorize(lambda i: valcols[i])
    newdata["largest"] = data[valcols].max(axis=1)
    newdata["pred"] = selval(data[valcols].values.argmax(axis=1))
    return newdata

def add_infiltration(data, cases):
    labels = list(data.index)
    found = []
    for case in cases:
        if case.id in labels:
            found.append(case)
    found.sort(key=lambda c: labels.index(c.id))
    infiltrations = [c.infiltration for c in found]
    data["infiltration"] = infiltrations
    return data

def modify_groups(data, mapping):
    """Change the cohort composition according to the given
    cohort composition."""
    data["group"] = data["group"].apply(lambda g: mapping.get(g, g))
    return data

def main():
    c_model = MLLDATA / "mll-sommaps/models/relunet_samplescaled_sommap_6class/model_0.h5"
    c_labels = MLLDATA / "mll-sommaps/output/relunet_samplescaled_sommap_6class/test_labels.json"
    c_preds = MLLDATA / "mll-sommaps/models/relunet_samplescaled_sommap_6class/predictions_0.csv"
    c_config = MLLDATA / "mll-sommaps/output/relunet_samplescaled_sommap_6class/config.json"
    c_cases = MLLDATA / "mll-flowdata/CLL-9F"
    c_sommaps = MLLDATA / "mll-sommaps/sample_maps/selected1_toroid_s32"
    c_misclass = MLLDATA / "mll-sommaps/misclassifications/"
    c_tube = [1, 2]

    # load datasets
    somdataset = ad.SOMDataset.from_path(c_sommaps)
    cases = cc.CaseCollection.from_dir(c_cases)

    # filter datasets
    test_labels = utils.load_json(c_labels)

    filtered_cases = cases.filter(labels=test_labels)
    somdataset.data[1] = somdataset.data[1].loc[test_labels, :]

    dataset = ad.CombinedDataset(
        filtered_cases, {ad.Datasets.from_str('SOM'): somdataset, ad.Datasets.from_str('FCS'): filtered_cases})

    # modify mapping
    config = utils.load_json(c_config)
    groupinfo = mappings.GROUP_MAPS[config["c_groupmap"]]
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

    dataset = loaders.DatasetSequence(
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

        #save gradients
        # class_string = "_".join(classes)
        # with open(f"{c_misclass}/{label}/{class_string}_gradients.p", "wb") as gradient_output:
        #     pickle.dump(gradients, gradient_output)

        for tube in c_tube:

            heatmaps = plotting.draw_saliency_heatmap(case, gradients, classes, tube)
            for idx,heatmap in enumerate(heatmaps):
                plotting.save_figure(heatmap,f"{c_misclass}/{label}/{classes[idx]}_tube_{tube}_saliency_heatmap.png")

            scatterplots = plotting.plot_tube(case, tube, gradients[tube - 1], classes=classes, sommappath=c_sommaps)
            for idx,scatterplot in enumerate(scatterplots):
                plotting.save_figure(scatterplot,f"{c_misclass}/{label}/{classes[idx]}_tube_{tube}_scatterplots.png")


if __name__ == "__main__":
    main()
