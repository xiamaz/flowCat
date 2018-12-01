Static data
===========

This directory contains static data, related to processing of the input data.

Currently contained are:

:code:`selected_cases.txt`
    Expert selected labels, which are representative for their respective
    cohorts

:code:`test_labels.json`
    Statically set random stratified split of 10% of the entire cohort. Does not
    contain any labels contained in the :code:`train_labels.json`.

:code:`train_labels.json`
    Statically set random group-stratified of 90% of the entire cohort.
