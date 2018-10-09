#!/bin/env python3
# pylint: skip-file
"""Test pregating combined with some visualization."""
from flowcat.visual import plotting
from flowcat.data import fcsdata
from flowcat.models import pregating


testfile = fcsdata.FCSData.from_path("/data/AWS/mll-flowdata/CLL-9F/AML/00b6627f9e2d56949ff4e51a6c686c0cb4696a12-4 CLL 9F 01 001.LMD")

# Pregating transformer
pregater = pregating.SOMGatingFilter()
pregated = pregater.transform(testfile.copy())

# Select indices that have been pregated for plotting
selection = [
    (testfile.data.drop(pregated.data.index).index, "grey", "ungated"),
    (pregated.data.index, "blue", "gated"),
]

# Create pregating plot using scatterplot overview of channels
figure = plotting.plot_scatterplot(testfile, 1, selections=selection)
plotting.save_figure(figure, "test.png")
