"""
Pipelines containing processing steps for flow data.

Each pipeline inherits from Merge, filling steps with different processing
steps.
"""
from sklearn.pipeline import Pipeline

from .base import Merge
from .fcs import ScatterFilter, MarkersTransform
from .som import SOMNodes
from .tfsom import SelfOrganizingMap
from .pregating import SOMGatingFilter


class ClusteringPipeline:
    """
    Operate on a TubeView level.
    """
    def __init__(self, main: str, prefit: str, pretrans: str, markers: list):
        self._main = main
        self._prefit = prefit
        self._pretrans = pretrans
        self._markers = markers

        self.model = None

    def fit(self, data: "TubeView") -> "ClusteringPipeline":
        self.model = Merge(
            transformer=self._get_main(self._main),
            eachfit=self._get_pre(self._prefit, self._markers),
            eachtrans=self._get_pre(self._pretrans, self._markers),
        )
        self.model.fit(data.data)
        return self

    def transform(self, data: "CaseView") -> "CaseView":
        data.data = self.model.transform(data.data)
        return data

    @staticmethod
    def _get_main(name: str, *_):
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

    @staticmethod
    def _get_pre(name: str, markers: list, *_):
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
