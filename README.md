flowCat: Automated classification of MFC data
=============================================

Organization and general discussion at:
[Trello](https://trello.com/b/Krk9nkPg/flowcat).

Discussion channel: [Mattermost](https://mm.meb.uni-bonn.de).

Running the main project
------------------------

1. `00a_dataset_prepare.py`
    
    Split a given dataset directory into a test and a training set. The
    splitting parameters are hardcoded in `preprocess_cases`.

    The default will generate a dataset split on the data 2018-06-30.

    The generated output are two new metadata json files as well as config
    files.

    Usage:
    
    ```{.sh}
    ./00a_dataset_prepare.py [-h] <PATH_TO_DATASET> <OUTPUT>
    ```

    Generated datsets in the output directory can be loaded into new python
    datasets.

    ```python
    flowcat.CaseCollection.from_path(path=<PATH_TO_DATASET>, metapath=<PATH_OUTPUT>)
    ```

2. `00b_select_ref_cases.py`

    A reference SOM needs to be created for transforming single cases on the
    basis of a selection of cases.

    The given script will output a list of case ids for a given dataset and
    given selection parameters. These directly map to the parameters for the
    `filter_reasons` function.

    ```{.sh}
    ./00b_select_ref_cases.py --input /data/flowdata --meta /data/flowdata/meta --groups CLL,normal test.json
    ```

    The metadata should point to a basename for both a meta.json and a optional
    meta_config.json.

3. `01a_generate_ref_som.py`

    Create a reference SOM using the previously generated test.json.

4. `01b_create_soms.py`

    Generate SOMs for all cases in the initial dataset.

5. `02_train_model.py`

    Train a CNN model with validation split.

6. `03a_create_soms_test.py`

    Generate test-set SOMs using the same reference.

7. `03b_test_model.py`

    Get prediction accuracy using the previously generated model.

Performance considerations
--------------------------

IO of input data can become the limiting factor, especially if the FCS
files are loaded on-demand from S3 for the FCS end-to-end model.
Consider downloading the whole dataset before running the classifier
itself. This can be done with aws-cli.

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

tensorflow SOM code taken from cgorman.
