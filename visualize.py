import pickle
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib import cm
from flowcat.dataset.fcs import FCSData
from scatter_specs import scatter_specs
import sys, os

# du = {'filepaths': filepaths,
#       # 'all_fcsdata': all_fcsdata,
#       'groups': groups,
#       'predictions': predictions,
#       'predicted_class_index': predicted_class_index,
#       'predicted_class': predicted_class,
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
filepaths = d['filepaths']
gradients = d['gradients']
som1channels = d['som_channels'][0]
som2channels = d['som_channels'][1]
ts = d['ts']
somnodes = d['somnodes']
print(somnodes)
print(somnodes[1].shape)

d1 = FCSData(filepaths[0]['fcs']['path'])
d1.drop_empty()
d1 = d1.scale()

d2 = FCSData(filepaths[1]['fcs']['path'])
d2.drop_empty()
d2 = d2.scale()

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
    fig = Figure(figsize=(12, 8), dpi=200)

    max_x_diag = 0
    max_y_diag = 0
    
    for diagx, diagy in scatter_specs[ent]:
        #print('d', diagx, diagy)
        max_x_diag = max(max_x_diag, diagx)
        max_y_diag = max(max_y_diag, diagy)

    for diagx, diagy in scatter_specs[ent]:
        scatter_dims = scatter_specs[ent][(diagx,diagy)]

        cha = scatter_dims['x_label']
        chb = scatter_dims['y_label']

        #cha = 'CD19-APCA750'
        #cha = 'SS INT LIN'
        #cha = 'FS INT LIN'

        #chb = 'SS INT LIN'
        #chb = 'CD79b-PC5.5'
        #chb = 'CD45-KrOr'

        tube, chaindex, chbindex = find_tube_for_channels(cha, chb)
        if tube is None:
            print('dropping scatter plot', cha, chb)
            continue

        ga = gradients[entIndex][tube-1].reshape(1156,11)[:,chaindex]
        gb = gradients[entIndex][tube-1].reshape(1156,11)[:,chbindex]

        #grads = np.maximum(ga, gb)
        grads = np.maximum(ga, gb)

        # gmean = grads.mean()
        # gstd = grads.std()
        # lim = gmean+4*gstd

        maxgrads = np.max(grads)

        def events_with_gradient_above_and_below(lower, upper):
            #lim = 0.5*grads.max()
            gr = grads.reshape(-1)
            hits = np.logical_and(gr>lower*maxgrads, gr<=upper*maxgrads)
            print('ratio som cells hit for lim range', lower, '...', upper,
                  '->', np.mean(hits))

            # iterate som cells with large gradient
            large_som_cell_indices = np.where(hits)[0]

            tt = None
            for i in large_som_cell_indices:
                so = somnodes[tube] == i
                if tt is None:
                    tt = so
                else:
                    tt = np.logical_or(tt, so)
            return tt

        dt = d1 if tube==1 else d2


        # diese try catches brauchen wir nicht,
        # wenn dt die richtigen daten sind...
        xs = None
        try:
            xs = dt.data[cha]
        except KeyError:
            print('channel not found', cha)
            continue
        ys = None
        try:
            ys = dt.data[chb]
        except KeyError:
            print('channel not found', chb)
            continue

        # print(xs.shape)
        # print(ys.shape)

        from matplotlib.colors import hsv_to_rgb
        colors = [
            hsv_to_rgb((0, 0.5, 0.7)),
            hsv_to_rgb((1/12, 0.5, 0.7)),
            hsv_to_rgb((1/6, 0.5, 0.7)),
        ]

        subplot_index = (max_y_diag+1)*diagx+diagy+1
        # print('si', (max_x_diag, max_y_diag), (diagx, diagy), subplot_index)
        ax = fig.add_subplot(max_x_diag+1,
                             max_y_diag+1,
                             subplot_index)
        marker_size = 1 #0.5
        hue = 5/6
        ax.scatter(xs, ys, s=marker_size, marker='.', color=hsv_to_rgb((hue, 0.2, 0.8)),
                   zorder=0)
        ns = 4
        limfs = np.linspace(0.2, 0.6, ns)
        sats = np.linspace(0.2, 0.8, ns)
        for ii in range(ns):
            limf = limfs[ii]
            limfupper = limfs[ii+1] if ii<ns-1 else 1
            sat = sats[ii]
            ev = events_with_gradient_above_and_below(limf, limfupper)
            try:
                ax.scatter(xs.loc[ev],
                           ys.loc[ev],
                           s=marker_size, marker='.',
                           #color=hsv_to_rgb((0., 0.5, 0.9-limf)))
                           #color=hsv_to_rgb((0.83, 0.5+limf, 0.8-limf/3))
                           #color=colors[limi],
                           #color=hsv_to_rgb((0., 0.2+limf, 0.8)))
                           color=hsv_to_rgb((hue, sat, 0.8)),
                           zorder=3+ii)
            except TypeError:
                print('scatter error with limf', limf)
                pass

        ax.set_xlabel(cha)
        ax.set_ylabel(chb)

        ax.set_xticklabels([])
        ax.set_yticklabels([])


        # axes = fig.add_subplot(1, 3, 2)
        # axes.imshow(grads)
        # axes = fig.add_subplot(1, 3, 3)
        # axes.imshow(grads > lim)

    #fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle(f"Scatterplots Tube {tube} Class {ent}")

    FigureCanvas(fig)
    figFilename = f'{anfrageDir}/scatter-{ent}.png'
    fig.savefig(figFilename)
    print(f'wrote {figFilename}')


'''

ouch,
file a ist nicht unbedingt tube1 und file b ist nicht unbedingt tube2!!!




Traceback (most recent call last):
  File "visualize.py", line 139, in <module>
    ax.scatter(xs.loc[ev],
  File "/home/ml/anaconda3/envs/py36/lib/python3.6/site-packages/pandas/core/indexing.py", line 1478, in __getitem__
    return self._getitem_axis(maybe_callable, axis=axis)
  File "/home/ml/anaconda3/envs/py36/lib/python3.6/site-packages/pandas/core/indexing.py", line 1911, in _getitem_axis
    self._validate_key(key, axis)
  File "/home/ml/anaconda3/envs/py36/lib/python3.6/site-packages/pandas/core/indexing.py", line 1790, in _validate_key
    error()
  File "/home/ml/anaconda3/envs/py36/lib/python3.6/site-packages/pandas/core/indexing.py", line 1781, in error
    raise TypeError("cannot use label indexing with a null "
TypeError: cannot use label indexing with a null key





for d in `(cd ~/hannes-saas/backend/work/; ls -1)`; do  (cd ~/flowCat; python3 visualize.py ~/hannes-saas/backend/work/$d/du.pickle) && viewnior /home/ml/hannes-saas/backend/work/*/scatter-*.png

'''                




