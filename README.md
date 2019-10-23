flowCat: Automated classification of MFC data
=============================================

Discussion channel: [Mattermost](https://mm.meb.uni-bonn.de).

This tool generated self-organizing maps from raw flow cytometry data stored
in fcs files. Afterwards the generated SOM weights are used to classify
diagnoses using a simple CNN.

The general processing steps are:

0. Data QC
1. SOM Transformation
2. Classification
3. Evaluation

Conda can be used to install all required dependencies for the project.

Hardware requirements: Nvidia GPU compatible with Cuda 10.0

Software requirements can be installed with [miniconda](https://docs.conda.io/en/latest/miniconda.html).

## Setup for development

Create a new environment from the environment file given in the repo. This will create a new conda environment named `flowcat`.

```sh
conda env create -f environment.yml

python3 setup.py develop
```

## Quick start

```sh
# split data into train and test dataset
./00_prepare_dataset.py --data <DATASET PATH> --meta <DATASET META> --output <DS DIR>

# generate reference som
./01_generate_soms.py --data <DATASET PATH> --meta <TRAIN META> \
                      --reference_ids <DS DIR>/references.json \
                      --output <SOM DIR>

# train classifier on transformed soms
./02_train_classifier.py --data <INDIV SOM> --meta <INDIV SOM>.json <MODEL DIR>
# generate results on test set
TODO
```

## 0. Data QC

Main reason is to ensure that input data contains all necessary channels in the
correct tubes for processing and the data has been sampled from the correct
material (currently only peripheral blood and bone marrow).

- `./00a_dataset_prepare.py` removed duplicated cases in cohorts and splits the
  dataset into a train and a test cohort, the test cohort data is only used in
  final evaluation

- `./00b_select_ref_cases.py` selects n cases from each cohort with some
  selection criteria, such as high infiltration. These are used for generating
  the reference SOM.

- `./00c_dataset_channel_plots.py` plots channel intensity plots for all
  cohorts.

`00a` and `00b` are needed for the next step.


## 1. SOM generation

Each fcs file from a patient is used to generate a single FCS file.

- `./01a_create_ref_som.py` created a SOM from a sample of data using random
  initialization. This is afterwards used to create individual soms.

- `./01b_create_soms.py` creates individual SOMs for all cases in the dataset.

## 2. Classification model training

Train a classifier on the generated SOM data to classify a property in the
cases.

- `./02_train_classifier.py` trains a new CNN. The train data is split again
  into the data used for training the model and for validating the model output.

## 3. Classification model evaluation

- `./03_predict_cases.py` generates predictions for new cases in the test
  cohort.


## AWS data

In order to work with data saved on AWS, use the following command to sync a
local folder with the remote content.

``` {.sh}
aws sync s3://mll-flowdata/<DATASET NAME> <DATASET NAME>
```

Documentation
-------------

Documentation is contained in `docs` in rst/sphinx format. To read the
documentation in html format simply run `make html` inside the docs
folder. Built html documentation can be found in `docs/_build/html`.
Navigate there to read them in your browser.

Contributions
-------------

tensorflow SOM code adapted from cgorman.
