# pylint: skip-file
# flake8: noqa
"""Create plots for channel occlusion data."""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")
from flowcat import io_functions, utils, mappings


data = utils.URLPath("output/0-final/model-analysis/occlusion")
output = utils.URLPath("output/0-final/model-analysis/occlusion/plots")
output.mkdir()

group_data = [(p.name.split("_")[0], io_functions.load_json(p)) for p in data.glob("*.json")]

group_tubes = [
    (
        group,
        tube,
        np.mean([t[2] for t in gdata if t[0] == tube]),
        np.sqrt(np.mean([np.power(t[3], 2) for t in gdata if t[0] == tube]))
    )
    for group, gdata in group_data for tube in ("1", "2", "3")
]
colors = sns.color_palette("Blues")

pos = np.arange(len(mappings.GROUPS))

fig, ax = plt.subplots()
ax.bar(
    [pos[mappings.GROUPS.index(g)] + (int(t) - 2) * 0.2 for g, t, _, _ in group_tubes],
    [m for _, _, m, _ in group_tubes],
    width=0.2,
    # yerr=[s for _, _, _, s in group_tubes],
    # color=[colors[mappings.GROUPS.index(g)] for g, t, _, _ in group_tubes]
    color=[colors[int(t)] for g, t, _, _ in group_tubes]
)
ax.set_xticks(np.arange(len(mappings.GROUPS)))
ax.set_xticklabels(mappings.GROUPS)
ax.set_ylabel("Average channel occlusion loss")
patches = [mpl.patches.Patch(color=colors[i], label=f"Tube {t}") for i, t in enumerate(("1", "2", "3"))]
ax.legend(handles=patches)
fig.savefig(str(output / "group_tube.png"), dpi=300)

group_channels = {
    group: {
        tube: [(m, g) for t, m, g, _ in gdata if t == tube]
        for tube in ("1", "2", "3")
    } for group, gdata in group_data
}
markers = {
    tube: [m for m, _ in group_channels["normal"][tube]]
    for tube in ("1", "2", "3")
}
sns.set_style("white")
fig, axes = plt.subplots(9, 3, figsize=(9, 10))
colors = sns.cubehelix_palette(len(mappings.GROUPS), rot=4, dark=0.30)
colors = [*colors[1:], colors[0]]
for i, group in enumerate(mappings.GROUPS):
    for j, tube in enumerate(markers):
        ax = axes[i, j]
        data = group_channels[group][tube]
        pos = np.arange(len(data))
        ax.bar(
            pos,
            [m for _, m in data],
            width=0.8,
            color=[colors[i]],
        )
        ax.set_ylim(0, 12)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels([m for m, _ in data], rotation="vertical")
        ax.set_ylabel(f"{group}\nLoss")

for tube, ax in zip(markers, axes[0, :].flatten()):
    ax.set_title(f"Tube {tube}")

for ax in axes[:-1, :].flatten():
    ax.set_xticklabels([])
for ax in axes[:, 1:].flatten():
    ax.set_yticklabels([])
    ax.set_ylabel("")

fig.tight_layout()
fig.savefig(str(output / f"test_channels.png"), dpi=300)
plt.close("all")
