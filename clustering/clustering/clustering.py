"""
Clustering main class

Handle transformation of specified tubes, with optional filtering of input
data and consensus SOM data.

Save input config for later loading.
"""
import logging
import os

from sklearn.pipeline import Pipeline

from .utils import load_json, create_stamp, get_file_path, put_file_path
from .transformation import base
from .transformation.fcs import ScatterFilter, MarkersTransform
from .transformation.tfsom import SelfOrganizingMap
from .transformation.som import SOMNodes
from .transformation.pregating import SOMGatingFilter
from .collection import CaseCollection


LOGGER = logging.getLogger(__name__)


def get_main(name: str, *_) -> Pipeline:
    """Build main pipeline components."""
    steps = []
    if name == "normal":
        steps.append(("clust", SelfOrganizingMap(m=10, n=10)))
    elif name == "gated":
        channels = ["CD45-KrOr", "SS INT LIN"]
        positions = ["+", "-"]
        # first gate merged results in fitting and individual
        # cases in transformation
        steps.append(
            ("somgating", SOMGatingFilter(channels, positions))
        )
        steps.append(
            ("somcluster", SelfOrganizingMap(m=10, n=10))
        )
    else:
        raise RuntimeError("Unknown main transformation.")

    return Pipeline(
        steps=steps
    )


def get_pre(name: str, markers: list, *_) -> Pipeline:
    steps = []
    # basic preprocessing used in all preprocessings
    steps.append(
        ("scatter", ScatterFilter())
    )
    if name == "normal":
        # default preprocessing
        pass
    elif name == "som":
        steps.append(
            ("out", SOMNodes(20, 20, 512))
        )
    elif name == "gated":
        channels = ["CD45-KrOr", "SS INT LIN"]
        positions = ["+", "-"]
        steps.append(
            ("out", SOMGatingFilter(channels, positions))
        )
    elif name == "gatedsom":
        channels = ["CD45-KrOr", "SS INT LIN"]
        positions = ["+", "-"]
        steps.append(
            ("out", SOMGatingFilter(channels, positions))
        )
        steps.append(
            ("nodes", SOMNodes(20, 20, 512))
        )
    else:
        raise RuntimeError("Unknown preprocesing")

    steps.append(
        ("marker", MarkersTransform(markers))
    )

    return Pipeline(
        steps=steps
    )


class Clustering:
    """Transform input of case objects into histogram data."""

    def __init__(
            self,
            cases: "CaseCollection",
            train_opts: dict,
            transform_opts: dict,
            pipeline_opts: dict,
            output_path: str,
    ):
        self._pipelines = {}
        self._cases = cases
        self._pipeline_opts = pipeline_opts
        self._train_opts = train_opts
        self._transform_opts = transform_opts
        self._output_path = output_path

        self.train_view = self._cases.create_view(**self._train_opts)
        self.transform_view = self._cases.create_view(**self._transform_opts)

    @classmethod
    def from_args(
            cls,
            args: "Arguments",
    ):
        # case information for json inputs
        collection = CaseCollection(
            args.input, [int(t) for t in args.tubes.split(";")]
        )

        # selection options for all cases to be transformed
        transform_opts = {
            "labels": None,
            "num": args.upsampled if args.upsampled > 0 else None,
            "groups": list(
                map(lambda x: x.strip(), args.groups.split(";"))
            ) if args.groups else collection.groups,
        }

        # selection options for all cases to be trained
        train_opts = {
            "labels": [
                case
                for cases in load_json(get_file_path(args.refcases)).values()
                for case in cases
            ] if args.refcases else None,
            "num": args.num if args.num != -1 else None,
            "groups": [
                g for g in transform_opts["groups"]
                if g != "normal" or args.refnormal
            ]
        }

        # pipeline options
        pipeline_opts = {
            "main": args.pipeline,
            "prefit": args.prefit,
            "pretrans": args.pretrans,
        }

        # add timestamp to output directory
        outdir = "{}_{}".format(args.output, create_stamp())
        return cls(
            collection,
            train_opts, transform_opts, pipeline_opts,
            outdir
        )

    def fit_transform(self, tube: int):
        """Train som model on pipeline and save result."""

        # fit SOM model
        data = self.train_view.get_tube(tube)

        # load pipeline from a tube
        if tube not in self._pipelines:
            pipeline = base.Merge(
                transformer=get_main(self._pipeline_opts["main"]),
                eachfit=get_pre(
                    self._pipeline_opts["prefit"], data.markers
                ),
                eachtrans=get_pre(
                    self._pipeline_opts["pretrans"], data.markers
                ),
            )

            LOGGER.info("Fitting for tube %d", tube)
            pipeline.fit(data.data)
        else:
            pipeline = self._pipelines[tube]

        # transform new data on SOM model
        LOGGER.info("Transforming for tube %d", tube)
        trans_data = self.transform_view.get_tube(tube)

        trans_data.data = pipeline.transform(trans_data.data)

        outpath = os.path.join(self._output_path, "tube{}.csv".format(tube))
        put_file_path(outpath, trans_data.export_results().to_csv)

        # save pipeline into the pipeline dict
        self._pipelines[tube] = pipeline

    def run(self) -> None:
        """Transform all tubes"""
        for tube in self._cases.selected_tubes:
            self.fit_transform(tube)

    def load(self, path: str):
        """Load a Clustering model from a saved file.

        Load pipelines data, and use it for transformation purposes.
        """
        pass

    def save(self, path: str):
        """Save trained model to file. Keeping metadata on used cases for
        consensus SOM, channels needed etc.

        Save all pipelines for all tubes specified.
        """
        pass
