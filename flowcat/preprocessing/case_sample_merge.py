"""
Merge individual tubes for a given case with correct masking.
"""
from logging import getLogger
from sklearn.base import BaseEstimator, TransformerMixin

from flowcat.types.marker import Marker
from flowcat.types.fcsdata import join_fcs_data


LOGGER = getLogger(__name__)


def create_merge_marker(channel_data: "Union[Marker, str]") -> Marker:
    if isinstance(channel_data, Marker):
        return channel_data

    channel_data = Marker.name_to_marker(channel_data)
    if channel_data.color is not None and channel_data.antibody is not None:
        channel_data = channel_data.set_strict(True)
    return channel_data


class CaseSampleMergeTransformer(TransformerMixin, BaseEstimator):
    """Extract all matching Markers from the given case."""

    def __init__(self, channels: list = None):
        self._channels = [
            create_merge_marker(c) for c in channels
        ]

    @classmethod
    def from_json(cls, data: "List[Marker]") -> "CaseSampleMergeTransformer":
        return cls(data)

    def to_json(self):
        return self._channels

    def fit(self, X: "Case", *_) -> "CaseSampleMergeTransformer":
        if self._channels is not None:
            raise RuntimeError("Model has been already fitted to channels.")

        self._channels = list({c for s in X.samples for c in s.get_data().channels})
        return self

    def transform(self, X: "Case", *_) -> "FCSData":
        data = [sample.get_data() for sample in X.samples]
        joined = join_fcs_data(data, channels=self._channels)
        return joined
