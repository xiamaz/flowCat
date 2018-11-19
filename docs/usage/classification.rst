Classification of samples
=========================

Classification of a single sample, eg a single patient for which a complete
panel (split into multiple tubes) has been obtained, can be done with multiple
different classifiers, for which different intermediary prepresentations of the
raw FCS data might be necessary.

Models:
--------------------

Different models are available to be used for classification depending on the data representation available

* histogram: If you want to classify samples based on the histogram representation

* som: classify samples based on the generated som maps

* etefcs: classify samples directly from the fcs files


Run Classification:
-------------------------
To run the classification, configuration parameters need to be set. Some of the parameters that needs to be set are described below:

* Set the config parameter for input file path: *c_dataset_cases* in *classify_cases.py*

* Depending on the available intermediary representation set the config param for *c_dataset_paths * and selct the appropriate model by setting *c_model_name*

* Set how the dataset is to be split into test and training sets. A predefined list of labels for test and training sets can be specified by *c_split_train_labels* and *c_split_test_labels*

	* If a predefined list is not available, set *c_split_test_num* to specify the number of samples in each cohorot to be used for training. This generates a list of labels for training and test that can be used in subsequent runs

* Please check the remaining parameters in the script. These can be used to optimize the classifier as needed.

* Classification can be run by loading an exisiting configuration file. See *config_example.toml* for an example

Once the parameters are set, run classify_cases.py with