"""
Dicts containing different mappings.
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
            "LM": "LMF",
            "MCL": "MP",
            "PL": "MP",
        },
        "sizes": [2, 2, 3, 1, 1, ],
    },
    "3class": {
        "groups": ["CD5+", "CD5-", "normal"],
        "map": {
            "CLL": "CD5+",
            "MBL": "CD5+",
            "CM": "CD5+",
            "MCL": "CD5+",
            "PL": "CD5+",
            "MP": "CD5+",
            "MZL": "CD5-",
            "LPL": "CD5-",
            "FL": "CD5-",
            "LM": "CD5-",
            "LMF": "CD5-",
            "HCL": "CD5-",
        },
        "sizes": [4, 4, 1, ],
    }
}

PATHOLOGIC_NORMAL = {
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
        }
    }
}
