import keras
import pathlib
import pandas as pd
import json

import map_class
from saliency import dataset_from_config
from classification import plotting


def build_ensemble(modelpaths, datashape):
    xshapes, yshape = datashape
    # Load and check models used for ensembling
    models = []
    for i, path in enumerate(modelpaths):
        model = keras.models.load_model(path)
        model.name = f"ensmodel_{i}"
        assert model.input_shape == [(None, *s) for s in xshapes], "Input mismatch"
        assert model.output_shape == (None, yshape), "Output mismatch"
        models.append(model)

    # Create inputs for ensemble
    inputs = [keras.layers.Input(shape=xshape) for xshape in xshapes]
    model_trees = [m(inputs) for m in models]

    output = keras.layers.average(model_trees)
    model = keras.models.Model(inputs=inputs, outputs=output, name="ensemble")
    return model


def main():
    #ENSEMBLE CONFIGURATION
    # c_models = [
    #     "mll-sommaps/models/relunet_nonglobal_200epoch_sommap_8class/model_0.h5",
    #     "mll-sommaps/models/relunet_yesglobal_200epoch_rep02_sommap_8class/model_0.h5",
    #     "mll-sommaps/models/relunet_yesglobal_200epoch_rep01_sommap_8class/model_0.h5",
    # ]
    c_models = [
        "mll-sommaps/models/relunet_yesglobal_500epoch_sample_weighted1510_sommap_8class/model_0.h5",
        "mll-sommaps/models/relunet_yesglobal_500epoch_nosampleweight_sommap_8class/model_0.h5",
    ]
    c_config = "mll-sommaps/output/relunet_yesglobal_200epoch_rep02_sommap_8class/config.json"
    c_indata = "mll-sommaps/output/relunet_yesglobal_200epoch_rep02_sommap_8class/data_paths.p"
    c_labels = "data/test_labels.json"
    c_ensemble_model = "mll-sommaps/models"
    c_ensemble_output =  "mll-sommaps/output"
    c_groupmap = "8class"

    c_name = "relunet_ensemble_500ep"

    ensemble_config = locals()

    modelpath = pathlib.Path(c_ensemble_model, c_name)
    modelpath.mkdir(parents=True, exist_ok=True)
    outputpath = pathlib.Path(c_ensemble_output, c_name)
    outputpath.mkdir(parents=True, exist_ok=True)
    # load dataset
    dataset = dataset_from_config(c_indata, c_labels, c_config, batch_size=16)

    model = build_ensemble(c_models, dataset.shape)

    groups = map_class.GROUP_MAPS[c_groupmap]["groups"]
    group_map = map_class.GROUP_MAPS[c_groupmap]["map"]

    #save ensemble model and model graph
    model.save(modelpath / 'model_0.h5')
    keras.utils.plot_model(model, to_file=modelpath / 'model_0.png')

    #predict on training dataset and create metrics + conf mat
    pred_mat = model.predict_generator(dataset, workers=4, use_multiprocessing=True)

    pred_df = pd.DataFrame(
        pred_mat, columns=dataset.groups, index=dataset.labels)
    pred_df["correct"] = dataset.ylabels
    pred_df.to_csv(outputpath / "predictions_0.csv")

    for gname, groupstat in map_class.GROUP_MAPS.items():
        if len(groups) < len(groupstat["groups"]):
            continue

        conf, stats = map_class.create_metrics_from_pred(pred_df, mapping=groupstat["map"])
        plotting.plot_confusion_matrix(
            conf.values, groupstat["groups"], normalize=True,
            title=f"Confusion matrix (f1 {stats['weighted_f1']:.2f} mcc {stats['mcc']:.2f})",
            filename=outputpath / f"confusion_{gname}", dendroname=None)
        conf.to_csv(outputpath / f"confusion_{gname}.csv")
        with open(outputpath / f"stats_{gname}.json", "w") as jsfile:
            json.dump(stats, jsfile)


if __name__ == '__main__':
    main()
