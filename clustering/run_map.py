import sys
import pathlib

import logging

import pandas as pd
import networkx as nx
import scipy as sp

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from clustering.transformation import tfsom, fcs, pregating
from clustering.collection import CaseCollection


GROUPS = [
    # "CLL", "PL", "FL", "HCL", "LPL", "MBL", "MCL", "MZL", "normal"
    "CLL", "MCL", "MBL", "normal"
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


def simple_run(path, data, reference):
    """Very simple SOM run for tensorflow testing."""
    # only use data from the second tube
    tubedata = data.get_tube(2)

    marker_list = tubedata.data[0].markers

    def data_generator():
        for tdata in tubedata.data:
            lmddata = tdata.data[marker_list]
            yield lmddata

    print(reference)

    model = tfsom.TFSom(
        m=10,
        n=10,
        dim=len(marker_list),
        batch_size=10,
        max_epochs=5,
        reference=reference,
        initialization_method="reference",
    )

    print(len(tubedata.data))

    model.train(data_generator, num_inputs=len(tubedata.data))

    weights = model.output_weights
    print(weights.shape)
    # weights.to_csv("somweights.csv")


def case_to_map(path, data, references):
    """Transform cases to map representation of their data."""

    for tube in [1, 2]:
        tubedata = data.get_tube(tube)

        reference = references[tube]

        model = tfsom.SOMNodes(
            m=10, n=10, dim=reference.shape[0],
            batch_size=1,
            initialization_method="reference", reference=reference
        )

        for case in tubedata.data:
            print(f"Transforming {case.parent.id}")
            filename = f"{case.parent.id}_t{tube}.csv"

            filepath = path / filename

            tubeweights = model.fit_transform(case.data[reference.columns])
            tubeweights.to_csv(filepath)


def generate_reference(path, data):
    """Create and save reference som maps using consensus soms."""

    for tube in [1, 2]:
        tubedata = data.get_tube(tube)
        marker_list = tubedata.data[0].markers
        model = tfsom.TFSom(
            m=10,
            n=10,
            dim=len(marker_list),
            batch_size=5,
            max_epochs=20,
            initialization_method="random",
        )

        def generate_data():
            for tcase in tubedata.data:
                data = tcase.data[marker_list]
                yield data

        model.train(generate_data, num_inputs=len(tubedata.data))
        weights = model.output_weights

        path.mkdir(parents=True, exist_ok=True)
        df_weights = pd.DataFrame(weights, columns=marker_list)
        df_weights.to_csv(path / f"t{tube}.csv")


def main():
    configure_print_logging()

    cases = CaseCollection(inputpath="s3://mll-flowdata/CLL-9F", tubes=[1, 2])

    # with open("labels.txt") as fobj:
    #     selected = [l.strip() for l in fobj]

    # reference_cases = cases.create_view(labels=selected)
    # print(len(reference_cases.data))

    # refpath = pathlib.Path("sommaps/reference")
    # generate_reference(refpath, reference_cases)

    # return

    reference_weights = {
        t: pd.read_csv(f"sommaps/reference/t{t}.csv", index_col=0)
        for t in [1, 2]
    }

    mappath = pathlib.Path("sommaps/lotta")
    mappath.mkdir(parents=True, exist_ok=True)

    transdata = cases.create_view(num=200, groups=GROUPS)

    metadata = pd.DataFrame({
        "label": [c.id for c in transdata.data],
        "group": [c.group for c in transdata.data]
    })
    metadata.to_csv(f"{mappath}.csv")
    case_to_map(mappath, transdata, reference_weights)


if __name__ == "__main__":
    main()
