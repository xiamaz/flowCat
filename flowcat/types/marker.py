from dataclasses import dataclass, replace
from flowcat.constants import MARKER_NAME_MAP


@dataclass(frozen=True)
class Marker:
    """Color and antibody information for a given channel.
    """

    color: str
    antibody: str
    meta: "ChannelMeta" = None
    strict: bool = False  # used to set that comparisons with this marker should be strict

    @property
    def name(self):
        return str(self)

    def _match(self, other: "Marker") -> bool:
        if (self.color is not None and other.color is not None) and self.color != other.color:
            return False
        elif (self.antibody is not None and other.antibody is not None) and self.antibody != other.antibody:
            return False
        return True

    def _match_strict(self, other: "Marker") -> bool:
        return self.antibody == other.antibody and self.color == other.color

    def matches(self, other: "Union[Marker, str]") -> bool:
        """Check whether the given string or marker matches.

        Different from a normal comparison, None values are treated as don't cares.
        """
        if isinstance(other, str):
            other = self.name_to_marker(other)
        return self._match(other)

    def matches_strict(self, other: "Union[Marker, str]") -> bool:
        if isinstance(other, str):
            other = self.name_to_marker(other)
        return self._match_strict(other)

    def _set_attr(self, **kwargs) -> "Marker":
        return replace(self, **kwargs)

    @staticmethod
    def _split_name(name: str) -> "Tuple[str, str]":
        parts = name.replace("-", " ").split(" ")
        try:
            antibody, color = parts
        except ValueError:
            antibody, color = parts[0], None

        antibody = MARKER_NAME_MAP.get(antibody, antibody)
        return (antibody, color)

    def set_color(self, color: str) -> "Marker":
        return self._set_attr(color=color)

    def set_meta(self, meta: "ChannelMeta") -> "Marker":
        """Returns a new copy with changed meta value."""
        return self._set_attr(meta=meta)

    def set_name(self, name: str) -> "Marker":
        antibody, color = self._split_name(name)
        return self._set_attr(antibody=antibody, color=color)

    def __eq__(self, other):
        """Compare markers loosely."""
        if isinstance(other, str):
            other = self.name_to_marker(other)
        if self.strict or other.strict:
            return self._match_strict(other)
        return self._match(other)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "-".join(s for s in (self.antibody, self.color) if s is not None)

    @staticmethod
    def name_to_marker(name: str, meta: "ChannelMeta" = None) -> "Marker":
        """Parse the given marker name into a marker dataclass."""
        antibody, color = Marker._split_name(name)

        return Marker(color=color, antibody=antibody, meta=meta)

    @staticmethod
    def convert(data: "Union[str, Marker]") -> "Marker":
        """Automatically create marker if data is str."""
        if isinstance(data, str):
            data = Marker.name_to_marker(data)
        return data
