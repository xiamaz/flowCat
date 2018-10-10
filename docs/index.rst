.. flowCat documentation master file, created by
   sphinx-quickstart on Wed Oct 10 10:53:00 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to flowCat's documentation!
===================================

flowCat generates classifications on external variables for FCS data from flow
cytometry measurements on a per sample level. A number of labeled samples are
used as training, on the basis of which predictions for test samples can be
made.

.. toctree::
   :caption: Usage documentation
   :glob:
   :maxdepth: 1

   usage/installation.rst
   usage/inputdata.rst
   usage/somcreation.rst
   usage/classification.rst

The code is currently completely in python using especially tensorflow_ and
keras_ for classifications.

.. toctree::
   :caption: Developer documentation
   :glob:
   :maxdepth: 1

   devel/paths_io.rst

.. _tensorflow: https://www.tensorflow.org/api_docs/python/tf
.. _keras: https://keras.io



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
