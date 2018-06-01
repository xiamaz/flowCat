import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_table(
    "output/clustering/sans_pregating_selected_20180601_1124/tube1.csv",
    delimiter=";",
    index_col=0
)

sel_data = data[list(map(str, range(100)))]

groups, levels = pd.factorize(data["group"])

pca = PCA(n_components=3)
trans_data = pca.fit_transform(sel_data)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(trans_data[:, 0], trans_data[:, 1], trans_data[:, 2], c=groups)
plt.show()
plt.savefig("3dtube1")

kernel = KernelPCA(n_components=3, kernel="sigmoid")
trans_data = kernel.fit_transform(sel_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(trans_data[:, 0], trans_data[:, 1], trans_data[:, 2], c=groups)
plt.savefig("3dtube1_kernel_sigmoid")
plt.show()

trans_data = TSNE(n_components=2).fit_transform(sel_data)
fig = plt.figure()
ax = fig.add_subplot(111)
colors = [plt.cm.jet(float(i)/len(levels)) for i in range(len(levels))]
for i, group in enumerate(levels):
    gdata = trans_data[groups==i]
    sc = ax.scatter(gdata[:, 0], gdata[:, 1], c=colors[i], label=group)
ax.legend()
plt.savefig("3dtube1_tsne_2d")
plt.close()
