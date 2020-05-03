# pylint: skip-file
# flake8: noqa
import pathlib
from urllib.parse import urlparse

from argmagic import FunctionInformation


class URLPath(pathlib.PosixPath):
    __slots__ = (
        "_scheme",
        "_netloc"
    )

    @classmethod
    def _from_parts(cls, args, *other, **kwargs):
        first = args[0]
        if isinstance(first, URLPath):
            scheme = first._scheme
            netloc = first._netloc
            path = first
        else:
            urlpath = urlparse(str(first))
            scheme = urlpath.scheme
            netloc = urlpath.netloc
            path = urlpath.path

        self = super()._from_parts((path, *args[1:]), *other, **kwargs)
        self._scheme = scheme
        self._netloc = netloc
        return self

    def _from_parsed_parts(self, *args, **kwargs):
        obj = super()._from_parsed_parts(*args, **kwargs)
        obj._scheme = self._scheme
        obj._netloc = self._netloc
        return obj

    def mkdir(self, mode=0o777, exist_ok=True, parents=True):
        return super().mkdir(mode=mode, exist_ok=exist_ok, parents=parents)

    def open(self, mode="r", *args, **kwargs):
        if "w" in mode:
            self.parent.mkdir(exist_ok=True, parents=True)
        return super().open(mode=mode, *args, **kwargs)

    def __str__(self):
        if self._scheme:
            return f"{self._scheme}://{self._netloc}{super().__str__()}"
        else:
            return super().__str__()

    def __add__(self, other):
        return self.__class__(str(self) + str(other))

    def __radd__(self, other):
        return self.__class__(str(other) + str(self))


def has_urlpath_in_hint(hint):
    if hasattr(hint, "__origin__"):
        return URLPath in hint.__args__
    else:
        return hint is URLPath


def _cast(obj):
    if isinstance(obj, URLPath):
        return obj
    elif isinstance(obj, str):
        return URLPath(obj)
    raise TypeError(f"Cannot cast {type(obj)} to URLPath")


def cast_urlpath(fun):
    """
    Cast arguments marked as URLPath to URLPath.
    """
    info = FunctionInformation(fun)
    positions = [
        (i, name)
        for i, (name, par_info) in enumerate(info.args.items())
        if has_urlpath_in_hint(par_info.typehint)
    ]

    def wrapper(*args, **kwargs):
        len_args = len(args)
        for pos, name in positions:
            if pos < len_args:
                args = (*args[:pos], _cast(args[pos]), *args[pos+1:])
            elif name in kwargs:
                kwargs[name] = _cast(kwargs[name])

        return fun(*args, **kwargs)

    return wrapper
