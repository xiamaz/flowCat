"""
Approaches for selecting valid marker sets from datasets.

This is currently only used in some external scripts, but kept here for future
integration.
"""
from typing import List, Dict
import logging

import flowcat
import collections


LOGGER = logging.getLogger(__name__)


def print_marker_ratios(counts, num, tube):
    """Print selected marker ratios on all cases."""
    LOGGER.debug(f"Tube %s", tube)
    for marker, count in counts.items():
        LOGGER.debug(f"\t%s\t\t%s/%s (%s)", marker, count, num, count / num)


def get_selected_markers(
        cases: flowcat.CaseCollection,
        tubes: List[str],
        marker_threshold: float = 0.9) -> flowcat.CaseCollection:
    """Get a list of marker channels available in all tubes."""
    selected_markers = {}
    for tube in tubes:
        marker_counts = collections.Counter(
            marker for t in [case.get_tube(tube) for case in cases]
            if t is not None for marker in t.markers
        )

        if LOGGER.isEnabledFor(logging.INFO):
            print_marker_ratios(marker_counts, len(cases), tube)

        # get ratio of availability vs all cases
        selected_markers[tube] = [
            marker for marker, count in marker_counts.items()
            if count / len(cases) > marker_threshold and "nix" not in marker
        ]

    selected_cases, _ = cases.filter_reasons(selected_markers=selected_markers)
    selected_cases.selected_markers = selected_markers

    return selected_cases
