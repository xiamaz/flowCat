Creation of self-organizing maps
================================

`self-organizing maps`_ are used to transform raw FCS data into the defined
dimensions of nodes on the map.


.. _self-organizing maps: https://en.wikipedia.org/wiki/Self-organizing_map

SOM maps generation:
--------------------

* If you are using aws then directly run the create_sommaps.py without any changes

* If the data is stored locally, update the configuration parameter for input file path: *c_general_cases* in *create_sommaps.py*

Types of SOM maps:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* reference SOM map: A reference SOM map is used as intial reference for generation of individual SOMs

	* to generate reference set, run create_sommaps.py with `--references` option
	* ::
	
	    python create_sommaps.py --references
		
	
* individual SOM maps: Generate SOM map for individual sample

	* to generate individual SOM maps, run create_sommaps.py without any option
	* ::
	
	    python create_sommaps.py
	
	* The list of samples for which the maps are to be generated can be defined in `data/selected_labels.txt`. If you wish to generate maps for all the samples then set c_soms_labels(or c_reference_labels) to `None`
	
* Please look at the various configuration parameters in create_sommaps.py. These can be changed accordingly to generate SOM maps as needed

output:
^^^^^^^^^^
* The generated map is saved in /mll-sommaps/reference_maps/<modelname> (default loaction, can be changed) whereas the individual SOM maps are in /mll-sommaps/sample_maps/<modelname> folder

* The <modelname>.csv in the sample_maps/ folder contains the label and the class mapping



 