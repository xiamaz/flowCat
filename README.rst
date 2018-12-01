Classification of MFC data using self-organizing maps
=====================================================

Organization and general discussion at: Trello_.

.. _Trello: https://trello.com/b/Krk9nkPg/flowcat

Folder structure
----------------

preprocessing
    Old R and flowSOM based upsampling

clustering
    Transformation of FCS information into histrogram form with SOM

classification
    Prediction of cohort labels using histogram data

report
    Analysis and presentation of information in the other stages

aws
    Convenience scripts for job submission into AWS Batch


Installation
------------

The project requires python 3.6 or later and tensorflow.

AWS interaction is required for:
- automatic download of case data and clustering data from S3
- automatic upload of results from clustering and classification to S3

AWS access is expected to be set up per user using normal credentials. awscli
is required for AWS Batch interactions and some upload and download operations.

docker CE is required for building and running of images generated from
dockerfiles individually for clustering and classification.

Both clustering and classification require installation of some dependencies
defined in :code:`requirements.txt` in their respective directories using
:code:`pip`. This can be made easier by using virtualenvs.


Contributions
-------------

Logicle functions from fcm package by Jacob Frelinger licensed under simplified BSD
