import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.utils import plot_model


model = load_model("mll-sommaps/models/selected_lr05-001_planar_s32_100ep_batchnorm_wrapped/model_0.h5")

plot_model(model, to_file="selected_batchnorm.png")

with open("mll-sommaps/models/selected_lr05-001_planar_s32_100ep_batchnorm_wrapped/history_0.p", "rb") as f:
    history = pickle.load(f)

history_df = pd.DataFrame(history)
history_df.iloc[:, [0, 2]].plot()
