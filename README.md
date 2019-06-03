flowCat: Automated classification of MFC data
=============================================

Organization and general discussion at:
[Trello](https://trello.com/b/Krk9nkPg/flowcat).

Discussion channel: [Mattermost](https://mm.meb.uni-bonn.de).

Module structure
----------------

``` {.}
flowcat/
```

### Runnable scripts

#### 0x - main scripts

Main scripts to generate SOMs and run the classification on the entire dataset.

#### 1x - special cohorts

Work on special cohorts. Such as AML or MM.

#### 2x - som transformation comparisons

Some scripts for testing modifications on the SOM implementation, as well
as comparisons of the implementation to results obtained from flowSOM.

#### 3x - classifier comparisons

TODO. Does not yet exist

#### Other scripts

Run availble unittests with:

``` {.sh}
./test.py
```

Unittests require some data saved in a separate S3 bucket flowcat-test.

There is a helper script to make syncing this bucket a bit easier.

``` {.sh}
./update_tests.sh  # will show usage
./update_tests.sh down  # download all test data to tests/data
```

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
