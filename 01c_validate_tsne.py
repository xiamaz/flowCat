"""
Validate generated soms by creating a tsne plot using a random stratified
subsample of test data."""
from argmagic import argmagic
import flowcat


def create_tsne(metapath: flowcat.utils.URLPath, somdata: flowcat.utils.URLPath, plotdir: flowcat.utils.URLPath):
    """Generate tsne plots for a subsample of data.

    Args:
        metapath: Path to metadata json for cases.
        somdata: Path to generated soms for cases.
        plotdir: Path to output plots for data.
    """


if __name__ == "__main__":
    argmagic(create_tsne)
