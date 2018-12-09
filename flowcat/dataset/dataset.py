import enum

from . import case_dataset, som_dataset, histo_dataset


class Dataset(enum.Enum):
    HISTO = enum.auto()
    SOM = enum.auto()
    FCS = enum.auto()

    @classmethod
    def from_str(cls, name):
        """Get enum type from a string as a case-insensitive operation."""
        if isinstance(name, cls):
            return name
        name = name.upper()
        return cls[name]

    def get_class(self):
        """Get the associated class for the given value."""
        if self == self.HISTO:
            return histo_dataset.HistoDataset
        elif self == self.SOM:
            return som_dataset.SOMDataset
        elif self == self.FCS:
            return case_dataset.CaseCollection
        else:
            raise TypeError(f"Type {self} has no associated class.")
