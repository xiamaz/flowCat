Paths and Input/Output management
*********************************

Python has two ways of managing filepaths in the standard library with most
external libraries accepting both of these representations.

string paths
    Normal strings as paths will simply use special functions define in
    :py:mod:`os.path` for operations such as path concatenation, reference to
    parents etc.

pathlib paths
    :py:mod:`pathlib` defines :py:class:`pathlib.Path` objects for paths. These
    enable the modeling of real and abstract paths for respective OS path
    schemas, such as Windows and UNIX filepaths. They also enable path
    concatenation via the division operator. Most functions available in
    :py:mod:`os.path` have been relegated to methods on the path objects.

Both of these are sufficient for working on a single system with only local
resources. The latter has some advantages regarding clarity of type and
automatic management of Path types in the path object itself. Additionally
discouragement of manual string operations on paths can be beneificial to avoid
path errors.

Neither of these solve issues arising from intermixing URLs and local paths.
While URL (Universal Resource Locator) might look similar to normal paths on
UNIX systems, they are defined in RFC1738 (eg not a POSIX standard) and contain
additonal facilities such as a fragment and query parts, that are unknown to a
normal path. Fortunately the latter are in most cases not of relevance, if we
are only interested in getting remote static data, which all have well defined
names.

For the usecase of host and protocol independent data location, we would just
need to encapsulate the scheme, authority and path segments of a URL. A normal
local path could now be encoded in a very similar way with just distinct values
for the scheme and authority. This has been implemented via
:py:class`flowcat.utils.URLPath`.

Usage
=====

The :py:class:`flowcat.utils.URLPath` object can be created from strings
containing paths or URLs, Path objects or other URLPaths.

Import the utility class in flowCat containing URLPath and other functions for
File IO.::
   from flowcat import utils

URLPath objects can created just like regular pathlib Paths.::
   from flowcat.utils import URLPath
   p = URLPath("local/relative/path")
   print(p.scheme)
