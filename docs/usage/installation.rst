Setup and installation
**********************

flowCat is completely written in python with most dependencies easily
installable via pip. This guide will mostly focus on running tensorflow on GPU.

Requirements
============

Most of the code can be made to run on CPU, they will just be unbearably slow.

Hardware requirements:

- GPU with minimum compute capability of 4.0

  - this can be checked in this `list of GPUs`_ (named as latest CUDA API
    version)

- at least 16GB RAM

- at least 300GB HDD storage for entire dataset

All of the code logic is contained in the flowCat repository. Data is currently
contained in S3 Buckets. In order to directly download raw data and previously
generated SOMs etc, it is necessary to obtain access to these buckets.
But generating data directly from raw FCS can be done without access.

As a minimal setup you will need access to some raw FCS data with associated
metainformation as input.

.. _list of GPUs: https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units

Prior considerations
====================

Tensorflow and most of its enviroments will be most easily installable on Ubuntu
(especially 16.04 LTS), but flowCat requires python 3.6, thus we will still need
to build tensorflow manually. Alternative distributions like ArchLinux, might
have recent enough packages to avoid manual compiling.

This guide has only been tested against:

* Ubuntu 16.04 (Tesla P40)

Python 3.6 enviroment
=====================

Installation of the correct python version can be handled via miniconda_.

You can skip this step if your system :program:`python3` is 3.6. In this case
you could also just use virtualenvs_ to manage python dependencies, if you do
not want to install for your user or system-wide.

Create `environment with python 3.6`_:

.. code-block:: sh

   # any name will do, but we use tensorflow_p36 as an example
   conda create -n tensorflow_p36 python=3.6
   # activate the new environment
   conda activate tensorflow_p36

Alternative usage of virtualenvs_.

.. code-block:: sh

   # create a new venv in directory env in the project root
   python3 -m venv env
   . env/bin/activate

.. _miniconda: https://conda.io/miniconda.html
.. _environment with python 3.6: https://conda.io/docs/user-guide/tasks/manage-python.html
.. _virtualenvs: https://docs.python.org/3.6/library/venv.html

Nvidia CUDA and CUDNN
=====================

Installation of CUDA_ can be referred from the nvidia manual. Any recent CUDA_
version (>= 9.0) should work for our purposes. If your OS is rather old, you can
try to look at `older CUDA versions`_ for prepackaged versions.

Example installation on Ubuntu 16.04
------------------------------------

.. code-block:: sh

   CUDA_VERSION=9.0  # any more recent version will also do
   sudo apt-get update && sudo apt-get install cuda-$CUDA_VERSION

CUDNN_ and NCCL_ have to match the installed cuda version.

CUDNN_ can be installed from the nvidia machine learning repo as a deb package.

.. code-block:: sh

   CUDNN_VERSION="7.3.1.20-1+cuda$CUDA_VERSION"
   # add the nvidia machine learning repo
   NVIDIA_REPO=/etc/apt/sources.list.d/nvidia-ml.list
   sudo sh -c "echo \"deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /\" > $NVIDIA_REPO"
   sudo apt-get update && sudo apt-get install libcudnn7=$CUDNN_VERSION libcudnn7-dev=$CUDNN_VERSION


Registration for `nvidia developer account`_ is required for easy access to
prebuilt packages of NCCL_. It is free, but it does take time.

We will need the OS-agnostic NCCL_ library for building tensorflow, we will need
it for compiling tensorflow.

.. code-block:: sh

   # download nccl from somewhere
   NCCL_VERSION="2.3.5-2+cuda$CUDA_VERSION"
   nccl_file=nccl_${NCCL_VERSION}_x86_64.txz
   # copy and untar the archive, clean up afterwards
   sudo cp $nccl_file /usr/local
   sudo tar xvf /usr/local/$nccl_file -C /usr/local/
   # nccl library will be located in /usr/local/nccl-2.3, this will be needed
   # in the tensorflow config
   sudo mv /usr/local/$(basename $nccl_file .txz) /usr/local/nccl-2.3
   sudo rm /usr/local/$nccl_file

Some `ubuntu setup scripts`_ can be used for reference. They also contain
commands to setup nvidia-docker, which we are currently not using for running
flowCat.

Installation on openSUSE Leap 42.3
----------------------------------

openSUSE Leap 42.3 is the default version of openSUSE installed via PXE on IGSB
computers.

- CUDA 10.0 is not available, install 9.2 instead
- use OS-agnostic versions of cuDNN and NCCL
- use miniconda to get python3.6

Follow the nvidia documentation on openSUSE setup. Do not use packages for Leap
15.0, since they are incompatible.

.. _CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
.. _CUDNN: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
.. _NCCL: https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html
.. _older CUDA versions: https://developer.nvidia.com/cuda-toolkit-archive
.. _nvidia developer account: https://developer.nvidia.com/developer-program
.. _ubuntu setup scripts: https://gist.github.com/xiamaz/b148b5f1ecc68c85b5d34ea15868d73b

Tensorflow installation
=======================

We will need to build tensorflow in order to work CUDA versions of our choice,
also we will be able to turn on any available CPU optimizations, which can help
with performance.

Bazel installation
------------------

Tensorflow requires bazel. Installers can be downloaded from `bazel releases`_.

Sample installation on Linux. Distro-agnostic.

.. code-block:: sh

   BAZEL_VERSION=0.17.2   # set to bazel version you need
   wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64 -O bazel.sh
   # execute bazel installer
   sh bazel.sh

.. _bazel releases: https://github.com/bazelbuild/bazel/releases

Building tensorflow
-------------------

Follow the `compilation guide`_. When configuring keep in mind to check yes for
CUDA and give it the correct paths to the locations you installed the nvidia
deep learning libraries.

.. code-block:: sh

   # make sure we are in the correct env and have the correct python version
   python -V
   # Python 3.6.X

   # using conda you can check that you are in the correct env
   echo $CONDA_DEFAULT_ENV
   # tensorflow_p36

   git clone https://github.com/tensorflow/tensorflow.git
   cd tensorflow
   git checkout r1.12

   # answer yes to cuda and set compute capability accordingly to your card
   # set nccl to your installed version and point to its location in /usr/local
   ./configure

   bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
   # if builds are failing you can try to restrict parallel jobs with either:
   # --jobs NUM_JOBS_PARALLEL
   # --local_resources 2048 (used RAM),.5 (used CPU),1.0 (IO capability)

   # create the pip package
   ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

   # install the generated package
   pip install /tmp/tensorflow_pkg/*.whl

- ``--config=opt --config=cuda`` will need to passed to bazel build to get
  native compiled packages and gpu extensions reliably, configure will often not
  be sufficient

.. _compilation guide: https://www.tensorflow.org/install/source

Test tensorflow
---------------

Try testing tensorflow installation using few lines in the interactive python
console.

.. code-block:: python

    import tensorflow as tf
    a = tf.constant(1)
    sess = tf.Session()
    # should output 1
    sess.run(a)

Other dependencies
==================

A list of required dependencies for flowCat are in :file:`requirements.txt`.

Install dependencies using: (Make sure you are in the correct environment)

.. code-block:: sh

    pip install -r requirements.txt

All scripts starting with test can be used to test some basic features.
