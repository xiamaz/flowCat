# pylint: skip-file
# flake8: noqa
import pathlib
from urllib.parse import urlparse


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

    def open(self, *args, **kwargs):
        self.parent.mkdir(exist_ok=True, parents=True)
        return super().open(*args, **kwargs)

    def __str__(self):
        if self._scheme:
            return f"{self._scheme}://{self._netloc}{super().__str__()}"
        else:
            return super().__str__()

    def __add__(self, other):
        return self.__class__(str(self) + str(other))

    def __radd__(self, other):
        return self.__class__(str(other) + str(self))
