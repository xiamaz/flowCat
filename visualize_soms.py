import pickle
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib import cm, rcParams
from flowcat.dataset.fcs import FCSData
from scatter_specs import scatter_specs
import sys, os

# du = {'filepaths': filepaths,
#       # 'all_fcsdata': all_fcsdata,
#       'groups': groups,
#       'predictions': predictions,
#       'predicted_class_index': predicted_class_index,
#       'predicted_class': predicted_class,
#       'somweights': transformed,
#       'somnodes': somnodes,
#       'grad_groups': grad_groups,
#       'gradients': gradients,
#       'ts': ts,
#       'som_channels': [som1['channels'],
#                        som2['channels']]}

assert len(sys.argv) == 2
args = sys.argv[1:]
du_filename = args[0]
anfrageDir = os.path.dirname(du_filename)

print('reading', du_filename)
with open(du_filename, 'rb') as f:
    d = pickle.load(f)

predicted_class = d['predicted_class']
groups = d['groups']
gradients = d['gradients']
som1channels = d['som_channels'][0]
som2channels = d['som_channels'][1]
somweights = d['somweights']

print('som1channels', som1channels)
print('som2channels', som2channels)

def find_tube_for_channels(cha, chb):
    for tube, sch in [(1, som1channels), (2, som2channels)]:
        # search in som1channels AND som2channels!
        if cha in sch and chb in sch:
            print('found channels in tube', tube, cha, chb)
            chaindex = sch.index(cha)
            chbindex = sch.index(chb)
            print(cha, chaindex)
            print(chb, chbindex)
            return tube, chaindex, chbindex
    
    return None, None, None

for entIndex, ent in enumerate(groups):
    max_y_diag = 0
    max_x_diag = 0

    
    for diagy, diagx in scatter_specs[ent]:
        max_y_diag = max(max_y_diag, diagy)
        max_x_diag = max(max_x_diag, diagx)

    if len(scatter_specs[ent]) <= 9:
        figsize = (9, 8)
    else:
        figsize = (9, 10.84)

    rcParams['font.size']       = 8
    rcParams['font.family']     = 'sans-serif'
    rcParams['font.sans-serif'] = 'DejaVu Sans'

    fig = Figure(figsize=figsize, dpi=200)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.93, hspace=0.3)

    # first the SOMs themselves
    for diagy, diagx in scatter_specs[ent]:
        scatter_dims = scatter_specs[ent][(diagy,diagx)]
        cha = scatter_dims['x_label']
        chb = scatter_dims['y_label']
        tube, chaindex, chbindex = find_tube_for_channels(cha, chb)
        if tube is None:
            print('dropping plot', cha, chb)
            continue

        plane1 = somweights[tube-1][0][::,::,chaindex]
        plane2 = somweights[tube-1][0][::,::,chbindex]
        image = np.stack([plane1, plane2, np.zeros_like(plane1)], axis=2)

        subplot_index = (max_x_diag+1)*diagy+diagx+1
        ax = fig.add_subplot(max_y_diag+1,
                             max_x_diag+1,
                             subplot_index,
                             aspect='equal')
        ax.imshow(image, aspect='equal')
        ax.set_title(f'red: {cha}\ngreen: {chb}')

        ax.set_xticklabels([])
        ax.set_yticklabels([])

    FigureCanvas(fig)
    figFilename = f'{anfrageDir}/soms-{ent}.png'
    fig.savefig(figFilename)
    print(f'wrote {figFilename}')

    # then the gradients
    for diagy, diagx in scatter_specs[ent]:
        scatter_dims = scatter_specs[ent][(diagy,diagx)]
        cha = scatter_dims['x_label']
        chb = scatter_dims['y_label']
        tube, chaindex, chbindex = find_tube_for_channels(cha, chb)
        if tube is None:
            print('dropping plot', cha, chb)
            continue

        plane1 = gradients[entIndex][tube-1][::,::,chaindex]
        plane2 = gradients[entIndex][tube-1][::,::,chbindex]
        image = np.stack([plane1, plane2, np.zeros_like(plane1)], axis=2)

        subplot_index = (max_x_diag+1)*diagy+diagx+1
        ax = fig.add_subplot(max_y_diag+1,
                             max_x_diag+1,
                             subplot_index,
                             aspect='equal')
        ax.imshow(image, aspect='equal')
        ax.set_title(f'red: {cha}\ngreen: {chb}')

        ax.set_xticklabels([])
        ax.set_yticklabels([])

    FigureCanvas(fig)
    figFilename = f'{anfrageDir}/soms-saliency-{ent}.png'
    fig.savefig(figFilename)
    print(f'wrote {figFilename}')
