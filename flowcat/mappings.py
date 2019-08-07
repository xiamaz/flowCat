"""
Dicts containing different mappings.

GROUP_MAPS: Mapping information for merging different cohorts

Each map is a dictionary containing:
- groups key (providing a list of groups, the order of which is used in all output generation)
- map, which is a dictionary containing all name maps, keep in mind, that more
  global maps should also contain the names of other merged classes in order to
  apply these after another mapping
- sizes, relative sizes of groups, only used for visual purposes in the
  confusion matrix, in most cases this should translate into the number classes
  contained in each label
"""
import json
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


PUBLIC_ENUMS = {
    "Material": Material,
}


# probe materials allowed in further processing
ALLOWED_MATERIALS = (Material.PERIPHERAL_BLOOD, Material.BONE_MARROW)

# groups without usable infiltration values
NO_INFILTRATION = ("normal")

# mapping of some legacy cohort names in order to correctly import older
# formats
NAME_MAP = {
    "HZL": "HCL",
    "HZLv": "HCLv",
    "Mantel": "MCL",
    "Marginal": "MZL",
    "CLLPL": "PL"
}

# Marker channel mapping to equalize naming
MARKER_NAME_MAP = {
    "Kappa": "kappa",
    "Lambda": "lambda",
}

# name of main groups in defined order for plotting
GROUPS = [
    "CLL", "MBL", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"
]

# merged group mappings, most useful in classification
GROUP_MAPS = {
    "8class": {
        "groups": ["CM", "MCL", "PL", "LPL", "MZL", "FL", "HCL", "normal"],
        "map": {"CLL": "CM", "MBL": "CM"},
        "sizes": [2, 1, 1, 1, 1, 1, 1, 1, ],
    },
    "6class": {
        "groups": ["CM", "MP", "LM", "FL", "HCL", "normal"],
        "map": {
            "CLL": "CM",
            "MBL": "CM",
            "MZL": "LM",
            "LPL": "LM",
            "MCL": "MP",
            "PL": "MP",
        },
        "sizes": [2, 2, 2, 1, 1, 1, ],
    },
    "5class": {
        "groups": ["CM", "MP", "LMF", "HCL", "normal"],
        "map": {
            "CLL": "CM",
            "MBL": "CM",
            "MZL": "LMF",
            "LPL": "LMF",
            "FL": "LMF",
            "MCL": "MP",
            "PL": "MP",
            # merged classes
            "LM": "LMF",
        },
        "sizes": [2, 2, 3, 1, 1, ],
    },
    "3class": {
        "groups": ["CD5+", "CD5-", "normal"],
        "map": {
            "CLL": "CD5+",
            "MBL": "CD5+",
            "MCL": "CD5+",
            "PL": "CD5+",
            "MZL": "CD5-",
            "LPL": "CD5-",
            "FL": "CD5-",
            "HCL": "CD5-",
            # merged classes
            "CM": "CD5+",
            "MP": "CD5+",
            "LM": "CD5-",
            "LMF": "CD5-",
        },
        "sizes": [4, 4, 1, ],
    },
    "2class": {
        "groups": ["patho", "normal"],
        "map": {
            "CLL": "patho",
            "MBL": "patho",
            "MCL": "patho",
            "PL": "patho",
            "LPL": "patho",
            "MZL": "patho",
            "FL": "patho",
            "HCL": "patho",
            # merged classes
            "CM": "patho",
            "MP": "patho",
            "LM": "patho",
            "LMF": "patho",
            "CD5+": "patho",
            "CD5-": "patho",
        },
        "sizes": [1, 1],
    }
}


# Common channel configurations
CHANNEL_CONFIGS = {
    "CLL-9F": {
        1: [
            "FS INT LIN",
            "SS INT LIN",
            "FMC7-FITC",
            "CD10-PE",
            "IgM-ECD",
            "CD79b-PC5.5",
            "CD20-PC7",
            "CD23-APC",
            "CD19-APCA750",
            "CD5-PacBlue",
            "CD45-KrOr",
        ],
        2: [
            "FS INT LIN",
            "SS INT LIN",
            "Kappa-FITC",
            "Lambda-PE",
            "CD38-ECD",
            "CD25-PC5.5",
            "CD11c-PC7",
            "CD103-APC",
            "CD19-APCA750",
            "CD22-PacBlue",
            "CD45-KrOr",
        ],
        3: [
            "FS INT LIN",
            "SS INT LIN",
            "CD8-FITC",
            "CD4-PE",
            "CD3-ECD",
            "CD56-APC",
            "CD19-APCA750",
            "HLA-DR-PacBlue",
            "CD45-KrOr"
        ],
    }
}
