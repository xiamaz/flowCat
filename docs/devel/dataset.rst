Programmatical usage of case datasets
=====================================

Collections of single cases are organized inside the
:mod:`flowcat.data.case_dataset` module. This module enables filtering, summary
statistics and iteration on collections of cases. Besides simple filtering it
also enables enforcement of more complex rules, such as all cases having data of
the same material available for the required tubes.

:class:`flowcat.data.case_dataset.CaseIterable`
   Base class implementing basic functions for iterable objects containg cases
   in a list.

:class:`flowcat.data.case_dataset.CaseCollection`
   Class used to refer to *complete* datasets. It represents all available data.
   Any special requirements to data integrity should be represented as filtering
   actions. Keeping this class as close to the data on disk enables us to use
   this class to generate raw metrics on our data.

:class:`flowcat.data.case_dataset.CaseView`
   Filtered representation of our dataset.

Initialization of datasets
--------------------------

Datasets in remote locations (eg S3) and local folders should be instantiated as
new :class:`CaseCollection`.::

   from flowcat.data import case_dataset

   # the path should point to the directory containing the case_info.json
   # file, it should not point to the file itself
   cases = case_dataset.CaseCollection("s3://mll-flowdata/CLL-9F")

Filtering operations can be used to get specific data.::

   # calling filter will return a new CaseView
   normal_cases = cases.filter(groups=["normal"])

   # selected_tubes and selected_markers will always be set for CaseView objects
   print(normal_cases.selected_tubes())
   # [1, 2, 3]
   print(normal_cases.selected_markers())
   # {1: [ ... ], 2: [ ... ]}

   # multiple filtering operations can be called at the same time
   cll_normal = cases.filter(groups=["CLL", "normal"], num=100)
   len(cll_normal)
   # 200, since 'num' will always select from each cohort, if a cohort as fewer
   # cases, the entire cohort will still be included

Collections for specific tubes
------------------------------

Often it is necessary to work on data from specific tubes, such as for many
transforation and classification approaches.
:class:`flowcat.data.case_dataset.TubeView` represents a collection of many
:class:`flowcat.data.case.TubeSample` objects. All samples inside a single tube
view should have the same tube, since this object should always be generated
from a :class:`flowcat.data.case_dataset.CaseView` with a
:method:`CaseView.get_tube` call.::

   tube1_data = cll_normal.get_tube(1)

   print(tube1[0])
   # instead of Case objects, this object contains TubeSample objects
