Setup and installation
**********************

flowCat is completely written in python with most dependencies easily
installable via pip. This guide will mostly focus on running tensorflow on GPU.

Prior considerations
====================

Tensorflow and most of its enviroments will be most easily installable on Ubuntu
(especially 16.04 LTS), but flowCat requires python 3.6, thus we will still need
to build tensorflow manually. Alternative distributions like ArchLinux, might
have recent enough packages to avoid manual compiling.

This guide has only been tested against:

* Ubuntu 16.04 (Tesla P40)

Python enviroment
=================

Installation of the correct python version can be handled via miniconda.

https://conda.io/miniconda.html

Create environment with python 3.6:

https://conda.io/docs/user-guide/tasks/manage-python.html

Nvidia CUDA and CUDNN
=====================

Installation of CUDA and CUDNN can be referred from the nvidia manual.

Registration for nvidia developer account is required for easy access to some
prebuilt packages. It is free, but it does take time.

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

CUDNN and NCCL have to be installed for the matching CUDA version!

https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

Get the OS-agnostic one, we will need it for compiling tensorflow:

https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html

Tensorflow compilation
======================

Tensorflow requires bazel, so let us get bazel in the easiest way.

Download and execute the linux installer:

https://github.com/bazelbuild/bazel/releases

Follow the compilation guide. When configuring keep in mind to check yes for
CUDA and give it the correct paths to the locations you installed the nvidia
deep learning libraries.

https://www.tensorflow.org/install/source

Some caveats:

- ``--config=opt --config=cuda`` will need to passed to bazel build to get
  native compiled packages and gpu extensions reliably
- if bazel jobs are being killed try to limit the number of jobs running in
  parallel using ``--jobs N``

Test tensorflow
===============

Try testing tensorflow installation using few lines in the interactive console.

.. python::
    import tensorflow as tf
    a = tf.constant(1)
    sess = tf.Session()
    # should output 1
    sess.run(a)

Other dependencies
==================

A list of required dependencies for flowCat are in requirements.txt.

Install dependencies using: (Make sure you are in the correct environment)

.. sh::
    pip install -r requirements.txt

All scripts starting with test can be used to test some basic features.

More caveats
============

- make sure you actually have access to input data, either you have access to S3
  (then you could just let it download into a tmp location), or you will need to
  have all the data locally
  - you will need approx 300GB space to fit all raw and processed data
    comfortably
  - tmp path can be changed by flowcat.utils.TMP_PATH in your executing script
  - s3 locations: mll-flowdata (raw), mll-sommaps (processed)
  - raw data is required for running the som generation, processed data is
    required for the classification itself
