#!/usr/bin/env python3
"""Experiment numbers."""
from pathlib import Path
from argparse import ArgumentParser

from report.base import create_parser
from report.overview import Overview


def get_args():
    parser = create_parser()
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    view = Overview(args.indir)
    # view.show()
    outdir = args.outdir / "numbers"
    view.write(outdir)


if __name__ == "__main__":
    main()
