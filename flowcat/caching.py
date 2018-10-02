"""
Simple caches
"""
import pathlib
import hashlib
import functools
import pickle

CACHEDIR = "cache"


def args_hasher(*args, **kwargs):
    """Use at own discretion. Will simply concatenate all input args as
    strings to generate keys."""
    hasher = hashlib.blake2b()
    hashstr = "".join(str(a) for a in args) + "".join(str(k) + str(v) for k, v in kwargs.items())
    hasher.update(hashstr.encode())
    return hasher.hexdigest()


def disk_cache(fun):
    """Cache function results depending on arguments on disk in a cache directory."""

    cachepath = pathlib.Path(CACHEDIR) / fun.__name__
    cachepath.mkdir(parents=True, exist_ok=True)

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        hashed = args_hasher(*args, **kwargs)
        filepath = cachepath / hashed
        if filepath.exists():
            with open(str(filepath), "rb") as f:
                result = pickle.load(f)
        else:
            result = fun(*args, **kwargs)
            with open(str(filepath), "wb") as f:
                pickle.dump(result, f)
        return result

    return wrapper


def mem_cache(fun):
    """Cache function output inside the calling object."""

    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_cache"):
            self._cache = {}
        hashed = args_hasher(*args, **kwargs)
        if hashed in self._cache:
            result = self._cache[hashed]
        else:
            result = fun(self, *args, **kwargs)
            self._cache[hashed] = result
        return result

    return wrapper
