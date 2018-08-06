#!/usr/bin/env python3
"""Experiment numbers."""
from pathlib import Path
from argparse import ArgumentParser
from report.overview import Overview


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--inpath",
        help="Folder containing input data.",
        default="../output",
        type=Path
    )

    parser.add_argument(
        "--outdir",
        help="Output directory for plots and tables.",
        default="figures/numbers",
        type=Path
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    view = Overview(args.inpath)
    # view.show()
    view.write(args.outdir)


if __name__ == "__main__":
    main()
