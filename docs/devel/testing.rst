Testing flowCat
===============

In order to ensure the correct operation of especially randomization and
transformation functions. It is advised to create tests to ensure correct
operation.

Tests can be found in `tests` and are written using the `unittest` python
module. This is used in our case for both normal unittests and also integration
tests, which may include AWS and other networking functionality.

If possible, unittests should be self-contained without data IO. Since some
parts of the code are very hard to separate from their IO components it might be
necessary to directly load some test data.

All additional test data should **never** be pushed into the main repository,
but should always be uploaded into a reachable file hoster, such as AWS S3.

Hosting of testing data
-----------------------

New testing data should be uploaded into the `flowcat-test` bucket, which is to
be directly synced into the `tests/data` directory.


Execute the following command in the project root to download the test files
from S3.

.. code-block:: sh
   aws s3 sync s3://flowcat-test tests/data
