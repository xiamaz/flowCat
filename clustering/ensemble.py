import keras
import pathlib
import pandas as pd
import json

import map_class
from saliency import dataset_from_config
from classification import plotting


def ensemble(models):
    input_1 = keras.layers.Input(shape=models[0].input_shape[0][1:])
    input_2 = keras.layers.Input(shape=models[0].input_shape[1][1:])
    model_layers = [model([input_1,input_2]) for model in models]
    average_layer = keras.layers.Average()(model_layers)
    model = keras.models.Model(inputs = [input_1,input_2], outputs = average_layer, name='ensemble')
    return model


def main():
    #ENSEMBLE CONFIGURATION
    c_model1 = "mll-sommaps/models/smallernet_double_yesglobal_epochrand_sommap_8class/model_0.h5"
    #c_model2 = "mll-sommaps/models/smallernet_double_yesglobal_epochrand_mergeavg_sommap_8class/model_0.h5"
    c_model3 = "mll-sommaps/models/smallernet_double_yesglobal_epochrand_mergemult_sommap_8class/model_0.h5"
    #c_model4 = "mll-sommaps/models/convolutional_2x2filter_noregu_sommap_8class/model_0.h5"
    c_model5 = "mll-sommaps/models/convolutional_2x2filter_yesregu_epochrand_mergeavg_sommap_8class/model_0.h5"
    #c_model6 = "mll-sommaps/models/convolutional_2x2filter_yesregu_epochrand_mergemult_sommap_8class/model_0.h5"
    #c_model7 = "mll-sommaps/models/convolutional_2x2filter_yesregu_epochrand_sommap_8class/model_0.h5"

    model_paths = locals()

    c_indata = "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/data_paths.p"
    c_config = "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/config.json"
    c_labels = "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/test_labels.json"
    c_ensemble_model = "mll-sommaps/models/epochrand_ensemble"
    c_ensemble_output =  "mll-sommaps/output/epochrand_ensemble"
    c_groupmap = "8class"

    group_maps = {
        "8class": {
            "groups": ["CM", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"],
            "map": {"CLL": "CM", "MBL": "CM"}
        },
        "6class": {
            "groups": ["CM", "MP", "LM", "FL", "HCL", "normal"],
            "map": {
                "CLL": "CM",
                "MBL": "CM",
                "MZL": "LM",
                "LPL": "LM",
                "MCL": "MP",
                "PL": "MP",
            }
        },
        "5class": {
            "groups": ["CM", "MP", "LMF", "HCL", "normal"],
            "map": {
                "CLL": "CM",
                "MBL": "CM",
                "MZL": "LMF",
                "LPL": "LMF",
                "FL": "LMF",
                "LM": "LMF",
                "MCL": "MP",
                "PL": "MP",
            }
        }
    }
    groups = group_maps[c_groupmap]["groups"]
    group_map = group_maps[c_groupmap]["map"]

    ## load models and data
    models = [keras.models.load_model(model_paths[path]) for path in model_paths]

    # rename models to avoid value error
    for i in range(len(models)):
        models[i].name = "model_{}".format(i)

    dataset = dataset_from_config(c_indata, c_labels, c_config, batch_size=1)

    #construct ensemble model
    ensemble_model = ensemble(models)

    #save ensemble model and model graph
    ensemble_model.save(pathlib.Path(c_ensemble_model,'model_0.h5'))
    keras.utils.vis_utils.plot_model(ensemble_model, to_file=pathlib.Path(c_ensemble_model,'model_0.png'))

    #load ensemble model
    #ensemble_model = keras.models.load_model(pathlib.Path(c_ensemble_model,"model_0.h5"))

    #predict on training dataset and create metrics + conf mat
    pred_mat = ensemble_model.predict_generator(dataset, workers=4, use_multiprocessing=True)

    pred_df = pd.DataFrame(
        pred_mat, columns=dataset.groups, index=dataset.labels)
    pred_df["correct"] = dataset.ylabels
    pred_df.to_csv(pathlib.Path(c_ensemble_model,"predictions_0.csv"))

    for gname, groupstat in group_maps.items():
        # skip if our cohorts are larger
        if len(groups) < len(groupstat["groups"]):
            continue

        conf, stats = map_class.create_metrics_from_pred(pred_df, mapping=groupstat["map"])
        plotting.plot_confusion_matrix(
            conf.values, groupstat["groups"], normalize=True,
            title=f"Confusion matrix (f1 {stats['weighted_f1']:.2f} mcc {stats['mcc']:.2f})",
            filename=pathlib.Path(c_ensemble_output,f"confusion_{gname}"), dendroname=None)
        conf.to_csv(pathlib.Path(c_ensemble_output,f"confusion_{gname}.csv"))
        with open(pathlib.Path(c_ensemble_output,f"stats_{gname}.json"), "w") as jsfile:
            json.dump(stats, jsfile)













if __name__ == '__main__':
    main()
