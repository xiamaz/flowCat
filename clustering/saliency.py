import os
import json
import pickle

import keras
from keras.models import load_model

from vis.utils import utils
from vis.visualization import visualize_saliency, overlay
from vis.visualization import visualize_cam

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import numpy as np

import map_class
import consensus_cases
from map_class import reshape_dataframe
from map_class import CountLoader


def calculate_saliency(model, input, layer_idx, filter_indices, backprop_modifier='guided'):
    '''Calculates saliency gradients'''
    grads_sal = visualize_saliency(
        model, layer_idx, filter_indices, input_indices=[*range(len(input))], seed_input=input, backprop_modifier='guided')
    return grads_sal


def plot_saliency_heatmap(gradients, plot_path, input_types):
    '''Plots saliency heatmaps'''
    for i, gradient in enumerate(gradients):
        if input_types[i] == 'h':
            gradient = np.reshape(gradient, (10, 10))
        plt.imshow(gradient, cmap='jet')
        plt.savefig(os.path.join(plot_path + "saliency_heatmap_{}".format(i)))


def plot_RGB_overlay(inputs, gradients, plot_path, input_types):
    '''Plots RGB overlay plots'''
    for i, input in enumerate(inputs):
        if input_types[i] == 'h':
            continue
        transformed_df = np.squeeze(transform_data(
            input, columns=['SS INT LIN', 'CD45-KrOr', 'CD19-APCA750']), 0)
        plt.imshow(gradients[i], cmap='jet')
        plt.imshow(scale_data(transformed_df, 3), alpha=0.5)
        plt.savefig(os.path.join(plot_path + "RGB_overlay_{}".format(i)))


def kappa_lambda_plot(gradients, tube2, plot_path):
    '''Calculates and plots Kappa Lambda ratio'''
    try:
        tube2_k = np.squeeze(transform_data(tube2, columns=['Kappa-FITC']))
        tube2_l = np.squeeze(transform_data(tube2, columns=['Lambda-PE']))
    except KeyError as e:
        print(e)

    tube2_kl_ratio = np.divide(tube2_k, tube2_l)
    plt.imshow(scale_data(tube2_kl_ratio, 1), cmap='RdGy')
    plt.title("Kappa/Lambda Ratio")
    plt.savefig(os.path.join(plot_path + "Kappa_Lambda"))


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
        model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(model)

    #extract label from 2DMap path if both histogram and 2DMaps are used as input
    if set(['h','m']) == set(input_types):
        label = os.path.basename(input[input_types.index("m")]).split('_')[0]
    else:
        label = 0

    transformed_input = [transform_saliency_input(
        path, input_types[i], i + 1, label) for i, path in enumerate(input)]

    # compute saliency gradients
    sal_grads = calculate_saliency(
        model, transformed_input, layer_idx, filter_indices)

    #plotting
    # plot saliency heatmap
    if saliency_heatmap:
        plot_saliency_heatmap(sal_grads, plot_path, input_types)
    # plot RGB overlay plots
    if rgb_plot:
        plot_RGB_overlay(input, sal_grads, plot_path, input_types)
    # plot Kappa/Lambda ratio
    if kappa_lambda:
        # check if second tube in 2Dmap format exists
        tube2 = [i for i, n in enumerate(input_types) if n == 'm'][1]
        kappa_lambda_plot(input[tube2], plot_path)


def dataset_from_config(datapath, labelpath, configpath, batch_size):
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

    dataset = map_class.SOMMapDataset(
        data, xoutputs, batch_size=1, draw_method="sequential", groups=groupinfo["groups"])

    return dataset


def main():
    ## SALIENCY GENERATOR CONFIG
    c_indata = "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/data_paths.p"
    c_model = "mll-sommaps/models/smallernet_double_noglobal_sommap_8class/model_0.h5"
    # c_model = "mll-sommaps/models/smallernet_double_yesglobal_epochrand_sommap_8class/model_0.h5"
    c_config = "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/config.json"
    c_labels = "mll-sommaps/output/smallernet_double_yesglobal_epochrand_sommap_8class/test_labels.json"
    c_preds = "mll-sommaps/models/smallernet_double_yesglobal_epochrand_sommap_8class/predictions_0.csv"

    # visualize the last layer
    c_layer_idx = -1

    ## LOAD existing model, data info and correct input data configuration.
    model = keras.models.load_model(c_model)

    # modify model for saliency usage
    # model.layers[c_layer_idx].activation = keras.activations.linear
    # model = utils.apply_modifications(model)

    dataset = dataset_from_config(c_indata, c_labels, c_config, batch_size=1)
    preds = pd.read_csv(c_preds, index_col=0)
    preds = consensus_cases.add_correct_magnitude(preds)

    ## GENERATE Saliency values for the input data
    all_labels = dataset.labels
    selected_label = dataset.labels[0]
    corr_group = preds.loc[selected_label, "correct"]
    pred_group = preds.loc[selected_label, "pred"]
    corr_idx = dataset.groups.index(corr_group)
    pred_idx = dataset.groups.index(pred_group)
    xdata, ydata = dataset[0]

    result = model.predict(xdata)
    print(result)

    for data in xdata:
        print(data.shape)
    print(corr_idx)
    grads_sal = visualize_saliency(
        model, c_layer_idx, 2, seed_input=xdata, input_indices=[0, 1], backprop_modifier='guided')
    for grad in grads_sal:
        print(grad.max())
    return

    ## SAVE Saliency values for later usage

    ## PLOT Saliency based information

    # histgram input
    input1 = '/home/jakob/Documents/flowCAT/abstract_somgated_1_20180723_1217/tube1.csv'
    input2 = '/home/jakob/Documents/flowCAT/abstract_somgated_1_20180723_1217/tube2.csv'
    input3 = '/home/jakob/Documents/flowCAT/selected1_toroid_s32/ffff7d20c895165bbda3b7ac1a7249c483fa4f6b_t1.csv'
    input4 = '/home/jakob/Documents/flowCAT/selected1_toroid_s32/ffff7d20c895165bbda3b7ac1a7249c483fa4f6b_t2.csv'
    # compute gradients
    grads = generate_saliency_plots(model, [input1, input2, input3, input4], layer_idx=-1, filter_indices=2, input_types=[
                                    'h', 'h', 'm', 'm'], plot_path="../../saliency_plots/histomap/")


if __name__ == '__main__':
    main()
