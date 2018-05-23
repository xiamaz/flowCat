import math
import logging
import re
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import fcsparser
from clustering.clustering import create_pipeline, FCSLogTransform

logging.basicConfig(level=logging.DEBUG)

RE_CHANNEL = re.compile(r"\$P\d+S")

def print_meta(metadata):
    machine = metadata["$CYT"]
    date = metadata["$DATE"]
    starttime = metadata["$BTIM"]
    endtime = metadata["$ETIM"]
    print("{} -- {} {}->{}".format(machine, date, starttime, endtime))

    channels = [
        metadata[m[0]] for m in [RE_CHANNEL.match(k) for k in metadata] if m
    ]
    channelinfo = {}
    for i, channel in enumerate(channels):
        colrange = metadata["$P{}R".format(i+1)]
        colexpo = metadata["$P{}E".format(i+1)]
        f1, f2 = colexpo.split(",")
        channelinfo[channel] = {
            "range": float(colrange),
            "f1": float(f1),
            "f2": float(f2),
        }
    print(" Channels:\n   {}".format("\n   ".join(channels)))
    return channelinfo


def print_data(data):
    print(data)


def plot_data(path, data, channels):
    fig = Figure()
    FigureCanvas(fig)
    # autosize grid
    r = math.ceil(math.sqrt(len(channels) + 1))
    for i, (xchannel, ychannel) in enumerate(channels):
        ax = fig.add_subplot(r, r, i+1)
        ax.scatter(data[xchannel], data[ychannel], s=1, c=data["predictions"])
        ax.set_xlabel(xchannel)
        ax.set_ylabel(ychannel)
    cax = fig.add_subplot(r, r, len(channels)+1)
    colorbar = matplotlib.colorbar.ColorbarBase(cax, orientation="horizontal")
    fig.tight_layout()
    fig.savefig(path)


input_file = "data/18-000002-PB CLL 9F 01 N17 001.LMD"

meta, data = fcsparser.parse(input_file, encoding="latin-1")


channel_info = print_meta(meta)
print(channel_info)

# data = FCSLogTransform().transform(data)
# def logTransform(data):
#     transCols = [c for c in data.columns if "LIN" not in c]
#     for col, f in channel_info.items():
#         if f["f1"]:
#             data[col] = 10 ** (f["f1"] * data[col] / f["range"]) * f["f2"]
#     print(transCols)
#     print(data.min(axis=0), data.max(axis=0))
# 
#     data[transCols] = np.log10(data[transCols])
#     return data

# data = logTransform(data)

# sel_chans = ["SS INT LIN", "CD45-KrOr", "FS INT LIN"]
sel_chans = ["SS INT LIN", "CD45-KrOr"]

pipe = create_pipeline()
pipe.fit(data[sel_chans])
weights = pipe.named_steps["clust"].model.output_weights
df_weights = pd.DataFrame(weights)
df_weights.columns = sel_chans
df_weights["predictions"] = df_weights.index
print(df_weights)
predictions = pipe.predict(data[sel_chans])
data["predictions"] = predictions
print(data)

channels = [
    ("SS INT LIN", "FS INT LIN"),
    ("CD45-KrOr", "SS INT LIN"),
    ("CD19-APCA750", "SS INT LIN"),
]
plot_data("testout2", data, channels)

# plot_data("testnodes1", df_weights, channels)
