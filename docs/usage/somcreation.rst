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

TensorBoard Visualization
----------------------------------------

The SOM maps generated above can be visualized in TensorBoard. See `TensorBoard <https://www.tensorflow.org/guide/summaries_and_tensorboard>`_ for deatils.


* To visualize the graphs/images and summary recorded, tensorflow event file needs to be generated.
* To do so, set the `tensorborddir` paramter in create_sommaps.py to a directory where you want the event file to be saved.
* To visualize the information in the event file, start tensorbaord using
* ::

    tensorboard --logdir <path to logdir containing tensorflow events>
  
* Tensorboard is started on 6006 port. open a browser and launch localhost:6006 to view the tensorboard
* various summary statistics and map weights and parameters recorded can be visualized on the tensorboard. Below are examples from tensorboard for SOM map visualization

* Summary statistics: The below image shows the statistics recorded such as quantization error, learning radius and the topographic error. These can be used as a measure to adjust the learning and SOM parameters
.. image:: ../images/tb_scalar.png

* The Map generated can also be visualized as shown below. Here the figure shows the selected marker channels in a colour map. These colour maps can be used as indicative of the marker distribution for each cohort
.. image:: ../images/tb_images.png

* TensorBoard also provides the option to view the tensor graph. This provides information regarding the learning and the model parameters. Please see `TensorBoard Graph viz <https://www.tensorflow.org/guide/graph_viz>`_ 
.. image:: ../images/tb_graph.png

