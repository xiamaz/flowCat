#########################
Classification components
#########################

Usage
-----

Clustering can be run by directly running the clustering package, which will
require input of a large number of arguments.

.. code:: sh
   $ python3 -m clustering -h

To make management of configuration sets easier, make can be used to run the
clustering steps.

.. code:: sh
   $ make run  # directly upload results to s3
   $ make run-local  # save output locally and use ./tmp for downloading

Specific comparisons can benefit from variation in single settings, while
controlling other configuration settings. Thus makefile configurations can again
be grouped into collections. Both single makefiles and collections can be found
in :code:`./experiments`.

Different experiments can be run by changing the TEMPLATE and EXP variables.
Keep in mind, that EXP overwrites any variables specified in TEMPLATE.

.. code:: sh
   $ make run-local EXP=experiments/test.mk  # run test using default template
