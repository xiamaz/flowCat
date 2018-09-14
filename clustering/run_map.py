import sys
import time
import pathlib
import collections

import logging

import numpy as np
import pandas as pd
import networkx as nx
import scipy as sp

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# import somoclu

from clustering.transformation import tfsom, fcs, pregating
from clustering.collection import CaseCollection, CaseView
from clustering import utils


utils.TMP_PATH = "/home/zhao/tmp"

GROUPS = [
    "CLL", "PL", "FL", "HCL", "LPL", "MBL", "MCL", "MZL", "normal"
    # "CLL", "MCL", "MBL", "normal"
]


def configure_print_logging(rootname="clustering"):
    """Configure default logging for visual output to stdout."""
    rootlogger = logging.getLogger(rootname)
    rootlogger.setLevel(logging.INFO)
    formatter = logging.Formatter()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    rootlogger.addHandler(handler)


def data_augmentor(tubecases, augmentation_table):
    """Create a list of cases with replication numbers."""
    for case in tubecases:
        reps = augmentation_table.get(case.parent.group, 1)
        for r in range(reps):
            yield r, case


def create_simple_generator(tubecases):
    """Create generator yielding dataframes from list of tubecases."""

    def generate_data():
        for tcase in tubecases:
            data = tcase.data
            yield data

    return generate_data


def create_z_score_generator(tubecases):
    """Normalize channel information for mean and standard deviation."""

    def generate_z_score():
        for tcase in tubecases:
            data = tcase.data
            data.drop("nix-APCA700", axis=1, inplace=True)
            zscores = preprocessing.StandardScaler().fit_transform(data)
            min_maxed = preprocessing.MinMaxScaler().fit_transform(zscores)
            yield pd.DataFrame(min_maxed, columns=data.columns)

    return generate_z_score


def simple_run(cases):
    """Very simple SOM run for tensorflow testing."""

    num_cases = 5
    gridsize = 32
    max_epochs = 10

    # always load the same normal case for comparability
    with open(f"labels_normal{num_cases}.txt", "r") as f:
        sel_labels = [l.strip() for l in f]

    # data = cases.create_view(num=1, groups=GROUPS) # groups=["normal"], labels=simple_label)
    data = cases.create_view(num=num_cases, groups=["normal"], labels=sel_labels)

    # with open("labels_normal1.txt", "w") as f:
    #    f.writelines("\n".join([d.id for d in data]))

    # only use data from the second tube
    tubedata = data.get_tube(2)

    # load reference
    reference = pd.read_csv("sommaps_new/reference_ep10_s32_planar/t2.csv", index_col=0)
    marker_list = tubedata.data[0].markers if reference is None else reference.columns

    # reference = None
    model = tfsom.TFSom(
        m=gridsize,
        n=gridsize,
        channels=marker_list,
        batch_size=1,
        radius_cooling="exponential",
        learning_cooling="exponential",
        map_type="planar",
        node_distance="euclidean",
        max_epochs=max_epochs,
        initialization_method="random" if reference is None else "reference",
        reference=reference,
        tensorboard=True,
        tensorboard_dir=f'tensorboard_refit_refactor',
        model_name=f"remaptest_remap_n{num_cases}"
    )
    for result in model.fit_map(create_z_score_generator(tubedata.data)()):
        print(result)
    return

    for (lstart, lend) in [(0.4, 0.04), (0.8, 0.08), (0.8, 0.8), (1.0, 0.1)]:

        data_generator = create_z_score_generator(tubedata.data)

        model = tfsom.TFSom(
            m=gridsize,
            n=gridsize,
            channels=marker_list,
            batch_size=1,
            # initial_radius=20,
            # end_radius=1,
            initial_learning_rate=lstart,
            end_learning_rate=lend,
            radius_cooling="exponential",
            learning_cooling="exponential",
            map_type="planar",
            node_distance="euclidean",
            max_epochs=max_epochs,
            initialization_method="random" if reference is None else "reference",
            reference=reference,
            tensorboard=True,
            tensorboard_dir=f'tensorboard_refit_refactor',
            model_name=f"remaptest_l{lstart:3f}-{lend:.3f}_n{num_cases}"
        )
        model.train(
            data_generator(), num_inputs=len(data)  # len(tubedata.data)
        )
    return

    # metainfo
    weights = model.output_weights
    for data in data_generator():
        counts = model.map_to_histogram_distribution(data, relative=False)
        print(counts)


def generate_reference(refpath, data, gridsize=10, max_epochs=10, map_type="planar", name=""):
    """Create and save consensus som maps."""
    name = name if name else "reference"

    for tube in [1, 2]:
        tubedata = data.get_tube(tube)
        marker_list = [m for m in tubedata.data[0].markers if "nix" not in m]
        model = tfsom.TFSom(
            m=gridsize,
            n=gridsize,
            channels=marker_list,
            batch_size=1,
            end_radius=2,
            radius_cooling="exponential",
            learning_cooling="exponential",
            map_type=map_type,
            node_distance="euclidean",
            max_epochs=max_epochs,
            initialization_method="random",
            tensorboard=True,
            model_name=f"{name}_t{tube}",
            tensorboard_dir=f'tensorboard_wednesday',
        )

        generate_data = create_z_score_generator(tubedata.data)

        time_a = time.time()
        model.train(generate_data(), num_inputs=len(tubedata.data))
        time_b = time.time()
        print(f"Training time: {(time_b - time_a):.3}s")
        weights = model.output_weights

        df_weights = pd.DataFrame(weights, columns=marker_list)
        # save the reference weight to the specified location
        output_path = refpath / f"{name}_{model.config_tag}" / f"t{tube}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_weights.to_csv(output_path)


def case_to_map(path, data, references=None, gridsize=10, map_type="planar"):
    """Transform cases to map representation of their data."""
    for tube in [1, 2]:
        tubedata = data.get_tube(tube)

        reference = references[tube] if references is not None else None
        used_channels = list(reference.columns) if reference else tubedata[0].data.columns
        used_channels = [c for c in used_channels if "nix" not in c]

        model = tfsom.SOMNodes(
            m=gridsize, n=gridsize, channels=used_channels,
            batch_size=1,
            initial_learning_rate=0.5,
            end_learning_rate=0.1,
            initial_radius=None,
            end_radius=1,
            map_type=map_type,
            max_epochs=10,
            initialization_method="reference" if reference else "random",
            reference=reference,
            counts=True,
            random_subsample=True,
            tensorboard=True,
            model_name=f"fitmap_t{tube}",
            tensorboard_dir="tensorboard_retraining/test",
            fitmap_args={
                "max_epochs": 20,
                "initial_learn": 0.1,
                "end_learn": 0.01,
                "initial_radius": 32,
                "end_radius": 1,
            }
        )
        generate_data = create_z_score_generator(tubedata)

        tubelabels = [c.parent.id for c in tubedata]
        circ_buffer = collections.deque(maxlen=20)
        time_a = time.time()
        for label, result in zip(tubelabels, model.transform(generate_data())):
            time_b = time.time()
            print(f"Saving {label}")
            filename = f"{label}_t{tube}.csv"
            filepath = path / filename
            result.to_csv(filepath)
            time_d = time_b - time_a
            circ_buffer.append(time_d)
            print(f"Training time: {time_d}s Rolling avg: {np.mean(circ_buffer)}s")
            time_a = time_b


def main():
    gridsize = 64
    simplerun = False
    createref = False
    max_epochs = 10
    maptype = "toroid"

    configure_print_logging()

    cases = CaseCollection(inputpath="s3://mll-flowdata/CLL-9F", tubes=[1, 2])

    if simplerun:
        simple_run(cases)
        return

    # generate consensus reference and exit
    if createref:
        sel_group = ["MCL"]
        with open("data/selected_cases.txt") as fobj:
            selected = [l.strip() for l in fobj]

        reference_cases = cases.create_view(num=1, groups=sel_group, infiltration=10)
        # reference_cases = cases.create_view(num=1, infiltration=20)
        refpath = pathlib.Path("wednesday_test/reference_maps")
        generate_reference(
            refpath,
            reference_cases,
            gridsize=gridsize,
            max_epochs=max_epochs,
            map_type=maptype,
            name=f"sample_init_{''.join(sel_group)}_{reference_cases[0].id}")
        return

    # reference_weights = {
    #     t: pd.read_csv(f"mll-sommaps/reference_maps/selected5_s32_e10_mplanar_deuclidean/t{t}.csv", index_col=0)
    #     for t in [1, 2]
    # }
    reference_weights = None

    mappath = pathlib.Path(f"wednesday_test/sample_maps/random_fourfold_{maptype}_s{gridsize}")
    mappath.mkdir(parents=True, exist_ok=True)

    # ref_trans = list(pd.read_csv("sommaps_aws/sample_maps/initial_toroid_s32.csv", index_col=0)["label"])
    ref_trans = None

    transdata = cases.create_view(groups=GROUPS, labels=ref_trans, num=5)

    case_to_map(mappath, transdata, reference_weights, gridsize=gridsize, map_type=maptype)

    metadata = pd.DataFrame({
        "label": [c.id for c in transdata.data],
        "group": [c.group for c in transdata.data],
    })
    metadata.to_csv(f"{mappath}.csv")


if __name__ == "__main__":
    main()