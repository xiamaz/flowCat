#############################
Dataset information and usage
#############################

Basic overview over input data processed in flowCat and some usage pointers to
utilize these data classes for more general tasks.

Background
**********

Data information
================

flowCat is currently concerned with the analysis of flow cytometric data in a
hematological diagnostic context. As such, besides the actual FCS data, we will
also need some additional metadata to make sense of our inputs.

FCS data should always be from standardized panels, such will mostly split their
relevant markers across multiple tubes. Thus we will need to link the
information between an FCS file for a single tube with the basic sample or
diagnostic procedure it is associated to. Additionally we also want to make sure
to not include the same patient multiple times if they were presented multiple
times, ie for time series information to our classification process, thus we
will need to make that the sample can be reliably linked to single cases.

The :py:mod:`flowcat.data.case_dataset` provides objects for the management of
these associations. Also handling issues such as:

- different tubes for a patient are not from the same material
  - different tubes should be from the same sample, so they should also always
    have the same material
- panel marker composition is incompatible with the used analysis methods
  - some analysis methods require a static set of known channels to work with,
    thus we will need to make sure, that these actually exist in the data, that
    we are working with
- availability of all required tubes
  - currently we will only analyze samples, where all required data is
    available
  - cases with missing tubes will need to be excluded

Implementation details
======================

The abstraction of cases with additional metainformation containing multiple
tubes with their respective metainformation is implemented in
:py:mod:`flowcat.data.case`. Operations on multiple cases, including parsing of
metainformation from data repositories and search and filtering operations are
included in :py:mod:`flowcat.data.case_dataset`.

Loading data from a folder or remote location containing organized data with an
associated json metadata file should use
:py:class:`flowcat.data.case_dataset.CaseCollection` to automatically parse the
contained metadata json file. This object also contains filtering functions,
which will return :py:class:`flowcat.data.case_dataset.CaseView`, which are
subselections of the entire dataset filtered on specific properties.
