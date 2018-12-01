#!/usr/bin/env python3
"""Compare classification results on a number of metrics."""
from argparse import ArgumentParser

import pandas as pd

from report.prediction import Prediction
from report.base import create_parser


TRANSFORM_COMP = {
    "9-class normal": {
        "seta": ("abstract_single_groups", "normal", "random"),
        "setb": ("abstract_single_groups_sqrt", "normal", "random")
    },
    "9-class somgated": {
        "seta": ("abstract_single_groups", "somgated", "random"),
        "setb": ("abstract_single_groups_sqrt", "somgated", "random")
    },
    "6-class merged somgated": {
        "seta": ("abstract_merged_hzl", "somgated", "random"),
        "setb": ("abstract_merged_hzl_sqrt", "somgated", "random")
    },
    "CD5 - normal": {
        "seta": ("cd5_threeclass", "normal", "random"),
        "setb": ("cd5_threeclass_sqrt", "normal", "random")
    },
    "CD5 - somgated": {
        "seta": ("cd5_threeclass", "somcombined", "random"),
        "setb": ("cd5_threeclass_sqrt", "somcombined", "random")
    },
}


def create_comparison(comparisons, pred_obj, outdir, custom_cols=None):
    """Create comparison table"""
    result_data = {}
    for desc, comp in comparisons.items():
        result = pred_obj.compare(**comp, path=outdir)
        result_data[desc] = result

    result_df = pd.DataFrame.from_dict(result_data, orient="index")
    if custom_cols is not None:
        result_df.columns = custom_cols
    result_df.to_latex(outdir / "transform_comparisons.latex", escape=False)


def main():
    args = create_parser().parse_args()
    pred = Prediction(args.indir)
    outdir = args.outdir / "classification"

    # histogram tranformation table
    custom_cols = [
        "Acc normal", "\\textnumero normal",
        "Acc sqrt", "\\textnumero Acc",
        "p-Value"
    ]
    create_comparison(TRANSFORM_COMP, pred, outdir, custom_cols)


if __name__ == "__main__":
    main()
