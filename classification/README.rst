upsampling output classification
################################

Histogram distribution data from upsampling is used for classification into
cohort labels.

usage
-----

Classification is run by using the :code:`./classify.py` script, which accepts
many command line arguments for configuration.

Management of different set of command line arguments can be handled by using
the included makefile, which also includes rules for downloading of clustering
data and uploading of classification results.

.. code:: sh
   # directly running classification requires giving a specific tag
   $ ./classify.py test
   $ make run EXP=experiments/test.mk  # run the specified experiment setting

   # sets of clustering experiments can be processed sequentially by using the
   # included run file to execute all subexperiments in the given folder
   $ ./run.fish experiments/abstract_merged

   # the run script will also automatically upload generated results back to S3
   # this can also be done manually by using
   $ make upload TEMP=<template_name> EXP=<exp_name>
