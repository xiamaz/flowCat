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

NAME_MAP = {
    "HZL": "HCL",
    "HZLv": "HCLv",
    "Mantel": "MCL",
    "Marginal": "MZL",
    "CLLPL": "PL"
}

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
