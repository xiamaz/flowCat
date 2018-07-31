#fflake8: noqa
import os
import re

import numpy as np
import pandas as pd







def main():
    experiments = {
        ex: get_folders("output/classification", ex)
        for ex in
        [
            "initial_comp_selected_normal_dedup",
            "initial_comp_selected_indiv_pregating_dedup"
        ]
    }

    # create dict of experiment name and list of loaded dataframes
    predictions = {
        ex: {
            k: v for e in folders for k, v in load_predictions(e).items()
        }
        for ex, folders in experiments.items()
    }

    td = predictions[list(predictions.keys())[0]]

    for experiment, dataframes in predictions.items():
        plot_experiment(experiment, dataframes)
