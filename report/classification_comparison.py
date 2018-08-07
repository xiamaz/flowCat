#!/usr/bin/env python3
"""Compare classification results on a number of metrics."""
from argparse import ArgumentParser

import pandas as pd

from report.prediction import Prediction
from report.base import create_parser


COMPARISONS = {
    "normal vs sqrt_transformed": {
        "seta": ("exotic", "exotic", "random"),
        "setb": ("exotic_sqrt", "exotic_sqrt", "random")
    },
    "normal vs somgated": {
        "seta": ("abstract_single_groups", "normal_dedup", "random"),
        "setb": ("abstract_single_groups", "somgated", "random")
    },
}


def main():
    args = create_parser().parse_args()
    pred = Prediction(args.indir)
    outdir = args.outdir / "classification"

    result_data = {}
    for desc, comp in COMPARISONS.items():
        result = pred.compare(**comp, path=outdir)
        result_data[desc] = result

    result_df = pd.DataFrame.from_dict(result_data, orient="index")
    result_df.to_latex(outdir / "comparisons.latex")


if __name__ == "__main__":
    main()
