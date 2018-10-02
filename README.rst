flowCat: Automated classification of MFC data
=============================================

Organization and general discussion at: Trello_.

.. _Trello: https://trello.com/b/Krk9nkPg/flowcat

Module structure
----------------

.. code::
    flowcat/


Runnable scripts
~~~~~~~~~~~~~~~~

``./create_sommaps.py``
    Generate SOMmap references and individual SOM maps for single cases.

``./classify_cases.py``
    Classify single cases to diagnoses using either direct FCS data, histogram
    data or individual SOM data.

``./test_pregating.py``
    Pregate a single fcs file and plot the result in 2D scatterplots with visual
    highlighting of the selected cell-population.

``./test_sommaps.py``
    Test the generation of a SOM map on a single case for a selected tube.


Contributions
-------------

tensorflow SOM code taken from cgorman.
