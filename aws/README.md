# AWS Batch submission control scripts

Some utility functions for submission of experiments to AWS Batch.

Currently this has been extensively used for the clustering process and to
a lesser degree for the classification process. (Latter mostly utilizing CPU-
Instances out of pragmatic reasons)

## Basic usage

Some utility functions can be found in the `batch_control.fish` script, while
functions for own script writing can be found in `lib/run_batch.fish`.

New experiments should mostly follow the pattern laid out in `classification`
and `clustering`.
