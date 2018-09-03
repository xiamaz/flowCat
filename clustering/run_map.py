import sys
import pathlib

import logging

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
            # TODO: possibly needs min-max scaling to remove negative values
            yield pd.DataFrame(zscores, columns=data.columns)

    return generate_z_score


# def somoclu_simple(data):
#     """Very simple SOM run using somoclu instead of tensorflow as the core."""
#     gridsize = 30
# 
#     tubedata = data.get_tube(2)
# 
#     data_generator = create_z_score_generator(tubedata.data)
#     testdata = list(data_generator())[0]
# 
#     model = somoclu.Somoclu(
#         n_columns=gridsize, n_rows=gridsize,
#         kerneltype=1,
#         maptype="planar", gridtype="rectangular",
#         std_coeff=0.5,
#         initialization="random",
#         verbose=2
#     )
#     model.train(
#         testdata.values.astype("float32"), epochs = 1000,
#         radius0=0, radiusN=1, radiuscooling="linear",
#         scale0=0.1, scaleN=0.01, scalecooling="linear"
#     )


def simple_run(data):
    """Very simple SOM run for tensorflow testing."""
    # only use data from the second tube
    tubedata = data.get_tube(2)

    gridsize = 30
    max_epochs = 10

    marker_list = tubedata.data[0].markers

    data_generator = create_z_score_generator(tubedata.data)

    model = tfsom.TFSom(
        m=gridsize,
        n=gridsize,
        channels=marker_list,
        batch_size=1,
        radius_cooling="exponential",
        learning_cooling="exponential",
        map_type="toroid",
        node_distance="euclidean",
        max_epochs=max_epochs,
        initialization_method="random",
        tensorboard=True,
        tensorboard_dir=f'simple_run_toroid/tb_s{gridsize}_c{len(data)}_e{max_epochs}',
    )
    model.train(
        data_generator(), num_inputs=len(tubedata.data)
    )
    return

    # metainfo
    weights = model.output_weights
    for data in data_generator():
        counts = model.map_to_histogram_distribution(data, relative=False)
        print(counts)


def case_to_map(path, data, references, gridsize=10):
    """Transform cases to map representation of their data."""

    for tube in [1, 2]:
        tubedata = data.get_tube(tube)

        reference = references[tube]

        # TODO: decrease initial radius and learning rate
        model = tfsom.SOMNodes(
            m=gridsize, n=gridsize, channels=list(reference.columns),
            batch_size=1,
            max_epochs=50,
            initialization_method="reference", reference=reference,
            counts=True
        )

        for case in tubedata.data:
            print(f"Transforming {case.parent.id}")
            filename = f"{case.parent.id}_t{tube}.csv"

            filepath = path / filename

            tubeweights = model.fit_transform(case.data[reference.columns])
            tubeweights.to_csv(filepath)


def generate_reference(path, data, gridsize=10, max_epochs=100):
    """Create and save consensus som maps."""

    for tube in [1, 2]:
        tubedata = data.get_tube(tube)
        marker_list = tubedata.data[0].markers
        model = tfsom.TFSom(
            m=gridsize,
            n=gridsize,
            channels=marker_list,
            batch_size=1,
            max_epochs=max_epochs,
            initialization_method="random",
            tensorboard=True,
            tensorboard_dir=f"tensorboard_c{len(data)}_ep{max_epochs}",
            model_name=f"reference_t{tube}_s{gridsize}",
        )

        generate_data = create_simple_generator(tubedata.data)

        model.train(generate_data(), num_inputs=len(tubedata.data))
        weights = model.output_weights

        df_weights = pd.DataFrame(weights, columns=marker_list)
        df_weights.to_csv(path / f"t{tube}_s{gridsize}.csv")


def main():
    gridsize = 50
    simplerun = True
    createref = True
    max_epochs = 1000

    configure_print_logging()

    cases = CaseCollection(inputpath="s3://mll-flowdata/CLL-9F", tubes=[1, 2])


    if simplerun:
        # always load the same normal case for comparability
        with open("simple.txt", "r") as f:
            simple_label = [l.strip() for l in f]
        simple_cases = cases.create_view(num=1, groups=["normal"], labels=simple_label)

        simple_run(simple_cases)
        return

    # generate consensus reference and exit
    if createref:
        # with open("labels.txt") as fobj:
        #     selected = [l.strip() for l in fobj]

        # reference_cases = cases.create_view(labels=selected)
        reference_cases = cases.create_view(num=1, infiltration=20)

        refpath = pathlib.Path(f"sommaps/reference_ep{max_epochs}_s{gridsize}")
        refpath.mkdir(parents=True, exist_ok=True)
        generate_reference(refpath, reference_cases, gridsize=gridsize, max_epochs=max_epochs)
        return

    reference_weights = {
        t: pd.read_csv(f"sommaps/reference/t{t}_s{gridsize}.csv", index_col=0)
        for t in [1, 2]
    }

    mappath = pathlib.Path(f"sommaps/huge_s{gridsize}_counts")
    mappath.mkdir(parents=True, exist_ok=True)

    with open("trans_labels.txt") as fobj:
        ref_trans = [l.strip() for l in fobj]

    transdata = cases.create_view(num=1000, groups=GROUPS, labels=ref_trans)

    metadata = pd.DataFrame({
        "label": [c.id for c in transdata.data],
        "group": [c.group for c in transdata.data],
    })
    metadata.to_csv(f"{mappath}.csv")
    case_to_map(mappath, transdata, reference_weights, gridsize=gridsize)


if __name__ == "__main__":
    main()
