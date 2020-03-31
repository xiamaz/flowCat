import enum


class Material(enum.Enum):
    """Class containing material types. Abstracting the concept for
    easier consumption."""
    PERIPHERAL_BLOOD = 1
    BONE_MARROW = 2
    OTHER = 3

    @staticmethod
    def from_str(label: str) -> "Material":
        """Get material enum from string"""
        if label in ["1", "2", "3", "4", "5", "PB"]:
            return Material.PERIPHERAL_BLOOD
        if label == "KM":
            return Material.BONE_MARROW
        return Material.OTHER
