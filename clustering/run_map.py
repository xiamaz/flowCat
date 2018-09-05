import sys
import pathlib

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
            zscores = preprocessing.StandardScaler().fit_transform(data)
            min_maxed = preprocessing.MinMaxScaler().fit_transform(zscores)
            yield pd.DataFrame(min_maxed, columns=data.columns)

    return generate_z_score


def simple_run(cases):
    """Very simple SOM run for tensorflow testing."""

    num_cases = 50

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
    # reference = pd.read_csv("sommaps_new/reference_ep10_s32_planar/t2.csv", index_col=0)
    reference = None

    gridsize = 32
    max_epochs = 10

    marker_list = tubedata.data[0].markers if reference is None else reference.columns

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
            tensorboard_dir=f'tensorboard_metrics_learning',
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


def generate_reference(refpath, data, gridsize=10, max_epochs=100):
    """Create and save consensus som maps."""

    for tube in [1, 2]:
        tubedata = data.get_tube(tube)
        marker_list = tubedata.data[0].markers
        model = tfsom.TFSom(
            m=gridsize,
            n=gridsize,
            channels=marker_list,
            batch_size=1,
            end_radius=2,
            radius_cooling="exponential",
            learning_cooling="exponential",
            map_type="toroid",
            node_distance="euclidean",
            max_epochs=max_epochs,
            initialization_method="random",
            tensorboard=True,
            model_name=f"reference_t{tube}",
            tensorboard_dir=f'tensorboard_reference',
        )

        generate_data = create_z_score_generator(tubedata.data)

        model.train(generate_data(), num_inputs=len(tubedata.data))
        weights = model.output_weights

        df_weights = pd.DataFrame(weights, columns=marker_list)
        # save the reference weight to the specified location
        output_path = refpath / model.config_tag / f"t{tube}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_weights.to_csv(output_path)


def case_to_map(path, data, references, gridsize=10):
    """Transform cases to map representation of their data."""

    for tube in [1, 2]:
        tubedata = data.get_tube(tube)

        reference = references[tube]

        model = tfsom.SOMNodes(
            m=gridsize, n=gridsize, channels=list(reference.columns),
            batch_size=1,
            initial_learning_rate=0.05,
            end_learning_rate=0.01,
            initial_radius=1,
            end_radius=1,
            map_type="toroid",
            max_epochs=5,
            initialization_method="reference", reference=reference,
            counts=True
        )

        for case in tubedata.data:
            print(f"Transforming {case.parent.id}")
            filename = f"{case.parent.id}_t{tube}.csv"

            filepath = path / filename

            tubeweights = model.fit_transform(case.data[reference.columns])
            tubeweights.to_csv(filepath)


def main():
    gridsize = 32
    simplerun = True
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
        with open("labels.txt") as fobj:
            selected = [l.strip() for l in fobj]

        reference_cases = cases.create_view(labels=selected)
        # reference_cases = cases.create_view(num=1, infiltration=20)
        refpath = pathlib.Path(f"sommaps_new/references")
        generate_reference(refpath, reference_cases, gridsize=gridsize, max_epochs=max_epochs)
        return

    reference_weights = {
        t: pd.read_csv(f"sommaps_new/reference_ep10_s{gridsize}_{maptype}/t{t}.csv", index_col=0)
        for t in [1, 2]
    }

    mappath = pathlib.Path(f"sommaps_new/test_{maptype}_s{gridsize}")
    mappath.mkdir(parents=True, exist_ok=True)

    # with open("trans_labels.txt") as fobj:
    #     ref_trans = [l.strip() for l in fobj]
    ref_trans = None

    transdata = cases.create_view(num=5, groups=GROUPS, labels=ref_trans)

    case_to_map(mappath, transdata, reference_weights, gridsize=gridsize)

    metadata = pd.DataFrame({
        "label": [c.id for c in transdata.data],
        "group": [c.group for c in transdata.data],
    })
    metadata.to_csv(f"{mappath}.csv")


if __name__ == "__main__":
    main()
