"""Script for generating Activation Maximization plots"""
import os
import pathlib
import keras

import numpy as np
import pandas as pd

from flowcat.visual import plotting
from flowcat import mappings
from flowcat.dataset import som_dataset
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import keras
from vis import visualization
from vis import utils
import keras.backend as K
from matplotlib.figure import Figure


if "MLLDATA" in os.environ:
    MLLDATA = pathlib.Path(os.environ["MLLDATA"])
else:
    MLLDATA = pathlib.Path('data')

def main():
    c_model = MLLDATA / "mll-sommaps/models/relunet_samplescaled_sommap_6class/model_0.h5"
    c_misclass = MLLDATA / "mll-sommaps/misclassifications/"
    c_tube = [1, 2]

    # load existing model
    model = keras.models.load_model(c_model)

    # modify model for saliency usage
    model.layers[-1].activation = keras.activations.linear
    model = utils.utils.apply_modifications(model)

    #visualize activation maximization plots for all classes
    #this is model dependent
    classes = mappings.GROUP_MAPS["6class"]["groups"]
    for cla in classes:
        #index of class that is going to be visualized
        filter_index = classes.index(cla)

        #compute optimal input for the class
        images = visualization.visualize_activation(model, layer_idx = -1, filter_indices = filter_index, input_indices=[0,1], input_range=(0,1000),act_max_weight=1,lp_norm_weight=0, tv_weight=0)

        #plot optimal input for both tubes
        for tube in c_tube:
            print(tube)
            #transform images to sommap data exluding the counts column
            img = images[tube-1]
            sommap_data = pd.DataFrame(data=img.reshape(1156, 12)[:,:-1],columns=[mappings.CHANNEL_CONFIGS["CLL-9F"][tube]])
            fig = plotting.plot_scatterplot(sommap_data,tube)
            FigureCanvas(fig)
            fig.savefig(f"{c_misclass}/AM_{cla}_{tube}.png")


if __name__ == "__main__":
    main()
