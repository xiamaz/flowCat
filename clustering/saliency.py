from keras.models import load_model
from keras import activations

from vis.utils import utils
from vis.visualization import visualize_saliency, overlay
from vis.visualization import visualize_cam
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import numpy as np
import os

from map_class import reshape_dataframe
from map_class import CountLoader


def transform_saliency_input(path: str, type: str, tube: int, label = 0, gridsize: int = 32, pad_width=0, columns=[]):
    '''Outputs specifically formatted data based on input type'''
    if type == 'h':
        if label != 0:
            return CountLoader.read_dataframe(path, tube=tube).loc[label]
        else:
            return CountLoader.read_dataframe(path, tube=tube).iloc[0]
    elif type == 'm':
        return transform_map_input(path, tube, gridsize, pad_width, columns)


def transform_map_input(path, tube, gridsize, pad_width, columns):
    map_list = []
    mapdata = pd.read_csv(path, index_col=0)
    if columns:
        mapdata = mapdata[columns]
    data = reshape_dataframe(
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


def main():

    # 2DMAP
    # load map2d model from hd5 file
    #model = load_model('/home/jakob/Documents/flowCAT/models/cllall1_planar_8class_60test_ep100/model_0.h5')
    # map2d input
    #input1 = '/home/jakob/Documents/flowCAT/cs-all1_planar_s32/0a1d2dfce8b39c9ef00c7dadc6279ce9a0c9579c_t1.csv'
    #input2 = '/home/jakob/Documents/flowCAT/cs-all1_planar_s32/0a1d2dfce8b39c9ef00c7dadc6279ce9a0c9579c_t2.csv'
    # compute gradients
    #grads = generate_saliency_plots(model, [input1, input2], layer_idx=-1, filter_indices=2, plot_path="../../saliency_plots/2DMap/", kappa_lambda_idx=1)

    # load histogram model from hdf5 file
    #model = load_model('/home/jakob/Documents/flowCAT/models/histogram/model_0.h5')
    # histgram input
    #input1 = '/home/jakob/Documents/flowCAT/abstract_somgated_1_20180723_1217/tube1.csv'
    #input2 = '/home/jakob/Documents/flowCAT/abstract_somgated_1_20180723_1217/tube2.csv'
    # compute gradients
    #grads = generate_saliency_plots(model, [input1,input2], layer_idx=-1,filter_indices=2, input_types = ['h','h'],plot_path="../../saliency_plots/histogram/")

    # Histomap
    model = load_model(
        '/home/jakob/Documents/flowCAT/models/histomap/model_0.h5')
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
