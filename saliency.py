import os
import json
import pickle

import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from vis.utils import utils
from vis.visualization import visualize_saliency, overlay
from vis.visualization import visualize_cam

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import numpy as np
import pathlib

from clustering import case_dataset as cc
import map_class
import consensus_cases
import vis_class


if "MLLDATA" in os.environ:
    MLLDATA = pathlib.Path(os.environ["MLLDATA"])
else:
    MLLDATA = pathlib.Path()


def plot_saliency_heatmap(gradients, classes, tube, plot_path):
    '''Plots saliency heatmaps'''
    if len(classes) > 1:
        for idx,group in enumerate(classes):
            plt.imshow(gradients[idx][tube-1].reshape(34,34), cmap='jet')
            plt.savefig(os.path.join(plot_path + f"saliency_heatmap_tube_{tube}_class_{group}"))
    else:
        plt.imshow(gradients[tube-1][0].reshape(34,34), cmap='jet')
        plt.savefig(os.path.join(plot_path + f"saliency_heatmap_tube_{tube}_class_{classes[0]}"))

def plot_RGB_overlay(input, gradients, classes, tube, plot_path):
    '''Plots RGB overlay plots'''
    transformed_df = np.squeeze(transform_map_input(input, columns=['CD45-KrOr','SS INT LIN', 'CD19-APCA750']), 0)
    if len(classes) > 1:
        for idx,group in enumerate(classes):
            plt.imshow(gradients[idx][tube-1].reshape(34,34), cmap='jet')
            plt.imshow(scale_data(transformed_df, 3), alpha=0.5)
            plt.savefig(os.path.join(plot_path + f"RGB_overlay_{tube}_class_{group}"))
    else:
        plt.imshow(gradients[tube-1][0].reshape(34,34), cmap='jet')
        plt.imshow(scale_data(transformed_df, 3), alpha=0.5)
        plt.savefig(os.path.join(plot_path + f"RGB_overlay_tube_{tube}_class_{classes[0]}"))

def kappa_lambda_plot(gradients, tube2, plot_path):
    '''Calculates and plots Kappa Lambda ratio'''
    try:
        tube2_k = np.squeeze(transform_map_input(tube2, columns=['Kappa-FITC']))
        tube2_l = np.squeeze(transform_map_input(tube2, columns=['Lambda-PE']))
    except KeyError as e:
        print(e)

    tube2_kl_ratio = np.divide(tube2_k, tube2_l)
    plt.imshow(scale_data(tube2_kl_ratio, 1), cmap='RdGy')
    plt.title("Kappa/Lambda Ratio")
    plt.savefig(os.path.join(plot_path + "Kappa_Lambda"))



def transform_map_input(path, columns, gridsize = 32, pad_width = 1 ):
    map_list = []
    mapdata = pd.read_csv(path, index_col=0)
    if columns:
        mapdata = mapdata[columns]
    data = map_class.reshape_dataframe(
        mapdata,
        m=gridsize,
        n=gridsize,
        pad_width=pad_width)
    map_list.append(data)
    return np.stack(map_list)


def scale_data(data, columns: int = 3):
    '''Transforms each channel of a numpy array to values between 0 and 1'''
    scaler = MinMaxScaler()
    if columns == 1:
        return scaler.fit_transform(data)
    else:
        for i in range(columns):
            data[:, :, i] = scaler.fit_transform(data[:, :, i])
        return data


def generate_saliency_plots(model, input, layer_idx, filter_indices, input_types, plot_path="plots", saliency_heatmap=True, rgb_plot=True, kappa_lambda=False):
    '''Generates for the given model and input multiple visualizations.

    Args:
        model: Path to an hdf5 file containing a keras model
        input: Path to input compatible with the model
        layer_idx: Index of the layer to be visualized.
        filter_indices: filter indices within the layer to be maximized. If None, all filters are visualized.
        input_type: Type of input. For each input the type has to be defined. Possible options: 'h' - histogram, 'm' - 2Dmap
        plot_path: Path were the generated plots will be saved.
        label: Sample label. This has to specified if histogram data and 2DMap are used as input.
        saliency_heatmap: Plot a saliency heatmap.
        kappa_lambda_idx: Plots the Kappa/Lambda ratio.
        rgb_plot: Plot the saliency heatmap with the channels SS,CD45 and CD19 as an RGB scaled overlay.
    '''
    if layer_idx == -1:
        # switch from softmax activation tif the last layer is visualized
        model.layers[layer_idx].activation = keras.activations.linear
        model = utils.apply_modifications(model)

    #extract label from 2DMap path if both histogram and 2DMaps are used as input
    if set(['h','m']) == set(input_types):
        label = os.path.basename(input[input_types.index("m")]).split('_')[0]
    else:
        label = 0

    #transformed_input = [transform_saliency_input(
    #    path, input_types[i], i + 1, label) for i, path in enumerate(input)]

    # compute saliency gradients
    sal_grads = visualize_saliency(model, layer_idx, filter_indices, seed_input=input, input_indices=[*range(len(input))])

    return sal_grads


def dataset_from_config(datapath, labelpath, configpath, batch_size,columns=None):
    """Load dataset from config specifications."""
    config = map_class.load_json(configpath)

    data = map_class.load_pickle(datapath)
    labels = map_class.load_json(labelpath)
    data = data.loc[labels, :]
    # modify mapping
    groupinfo = map_class.GROUP_MAPS[config["c_groupmap"]]
    data["orig_group"] = data["group"]
    data = map_class.modify_groups(data, groupinfo["map"])
    data = data.loc[data["group"].isin(groupinfo["groups"]), :]

    # config["c_dataoptions"]["Map2DLoader"]["sel_count"] = None

    modelfun, xoutputs, train_batch, test_batch = map_class.get_model_type(
        config["c_model"], config["c_dataoptions"], data)

    if columns != None:
        data = data[columns]

    dataset = map_class.SOMMapDataset(
        data, xoutputs, batch_size=1, draw_method="sequential", groups=groupinfo["groups"])

    return dataset

def main():
    ## SALIENCY GENERATOR CONFIG
    c_indata = MLLDATA / "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/data_paths.p"
    c_model = MLLDATA / "mll-sommaps/models/smallernet_double_yesglobal_epochrand_sommap_8class/model_0.h5"
    c_config = MLLDATA / "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/config.json"
    c_labels = MLLDATA / "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/test_labels.json"
    c_input1 = "mll-sommaps/sample_maps/selected1_toroid_s32/1230e71ad4d2d27a5ce6e0162335976a72300505_t1.csv"
    c_input2 = "mll-sommaps/sample_maps/selected1_toroid_s32/1230e71ad4d2d27a5ce6e0162335976a72300505_t2.csv"
    c_saliency = "mll-sommaps/saliency/"
    c_layer_idx = -1

    cases = cc.CaseCollection(MLLDATA / "mll-flowdata/CLL-9F", tubes=[1, 2])
    # model = keras.models.load_model("mll-sommaps/models/convolutional/model_0.h5")

    predictions = pd.read_csv(
        MLLDATA / "mll-sommaps/models/smallernet_double_yesglobal_epochrand_mergemult_sommap_8class/predictions_0.csv",
        index_col=0)

    #merged_predictions = merge_predictions(predictions, map_class.GROUP_MAPS["6class"])
    #result = map_class.create_metrics_from_pred(predictions, map_class.GROUP_MAPS["6class"]["map"])
    #print(result)

    predictions = vis_class.add_correct_magnitude(predictions)
    predictions = vis_class.add_infiltration(predictions, cases)

    correct, wrong = vis_class.split_correctness(predictions)

    false_negative = wrong.loc[wrong["pred"] == "normal", :]
    # print(false_negative.loc[:, ["infiltration", "correct", "largest"]])

    high_infil_false = false_negative.loc[false_negative["infiltration"] > 5.0, :]
    sel_high = high_infil_false.iloc[1, :]
    case = cases.get_label(sel_high.name)

    dataset = dataset_from_config(
        c_indata, c_labels, c_config, batch_size=1)
    all_labels = dataset.labels
    selected_label = dataset.labels[0]
    corr_group = predictions.loc[selected_label, "correct"]
    pred_group = predictions.loc[selected_label, "pred"]
    xdata, ydata = dataset.get_batch_by_label(case.id)
    print(case.id)

    model = keras.models.load_model(c_model)

    sal_grads = [generate_saliency_plots(model, xdata, c_layer_idx, dataset.groups.index(group), input_types=['m','m'],plot_path = c_saliency) for group in set([corr_group,pred_group])]

    #plotting
    for grads in sal_grads:
        # plot saliency heatmap
        plot_saliency_heatmap(grads, c_saliency)
        # plot RGB overlay plots
        plot_RGB_overlay([c_input1,c_input2], grads, c_saliency)
        # plot Kappa/Lambda ratio
        # if kappa_lambda:
        #     # check if second tube in 2Dmap format exists
        #     tube2 = [i for i, n in enumerate(input_types) if n == 'm'[1]
        #     kappa_lambda_plot(input[tube2], c_saliency)


if __name__ == '__main__':
    main()
