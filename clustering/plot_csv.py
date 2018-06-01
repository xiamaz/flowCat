import pandas as pd
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from clustering.plotting import plot_histogram

TUBE = 1
FOLDER = "sans_pregating_selected_20180601_1124"

data = pd.read_table("{}/tube{}.csv".format(FOLDER, TUBE), delimiter=";", index_col=0)

groups = data.groupby("group")
fig = Figure()
for i, (group, gdata) in enumerate(groups):
    mean_data = gdata.mean()
    std_data = gdata.std()
    ax = fig.add_subplot(5, 2, i+1)
    plot_histogram((mean_data, std_data), ax, title=group)

fig.set_size_inches(20, 10)
FigureCanvas(fig)
fig.suptitle("Tube {} histogram view.".format(TUBE))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("tube{}_{}".format(TUBE, FOLDER), dpi=100)
