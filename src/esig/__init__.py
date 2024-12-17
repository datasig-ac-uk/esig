#!/usr/bin/env python

#
# The ESig Python package
# Basic package functions
#

import functools
import os
import warnings
import math

import numpy


ESIG_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

from .backends import get_backend, set_backend, list_backends

# noinspection PyUnresolvedReferences
from pyrecombine import recombine

# noinspection PyUnresolvedReferences
__all__ = [
    "get_version",
    "is_library_loaded",
    "get_library_load_error",
    "stream2sig",
    "stream2logsig",
    "logsigdim",
    "sigdim",
    "sigkeys",
    "logsigkeys",
    "recombine",
    "get_backend",
    "set_backend",
    "list_backends",
    "backends",
    "__version__"
]

# with open(ESIG_PACKAGE_ROOT, "VERSION") as f:
#     __version__ = f.read().strip()

def get_version():
    """
    Returns the version number of the ESig package.
    The version is obtained from the VERSION file in the root of the package.

    Args:
        None

    Returns:
        string: The package version number. In format 'major.minor.release'.
    """
    warnings.warn("get_version is deprecated.", DeprecationWarning)
    return __version__

def is_library_loaded():
    """
    Determines whether the tosig Python extension can be successfully loaded.
    If the library cannot be loaded successfully, debugging information can be obtained from get_library_load_error().

    Args:
        None
    Returns:
        boolean: True iif the library can be loaded successfully; False otherwise.
    """
    warnings.warn("is_library_loaded is deprecated.", DeprecationWarning)

    return True


def get_library_load_error():
    """
    Returns a string containing the message of the exception raised when attempting to import tosig.
    If no exception is raised when attempting to import, None is returned.

    Args:
        None
    Returns:
        string: The message associated with the exception when attempting to import.
        None: If no exception is raised when importing, None is returned.
    """
    warnings.warn("get_library_load_error is deprecated.", DeprecationWarning)
    return None


def _verify_stream_arg(*types):
    """
    Helper decorator to provide type checking on the
    Numpy arrays
    """
    if types and callable(types[0]):
        fn = types[0]
        types = None
    else:
        fn = None

    types = types or (numpy.float32, numpy.float64)

    def decorator(func):

        @functools.wraps(func)
        def wrapper(stream, *args, **kwargs):
            as_array = numpy.array(stream)
            if not as_array.dtype in types:
                str_types = tuple(map(str, types))
                raise TypeError("Values must be of one of the following types {}".format(str_types))

            if not as_array.ndim == 2:
                raise ValueError("stream must be a 2-dimensional array")

            return func(as_array, *args, **kwargs)
        return wrapper

    if fn:
        return decorator(fn)
    return decorator


def _verify_valid_depth(width, depth):
    size = width ** (depth + 1) - 1
    if size > 2 ** 63 - 1:
        raise RuntimeError("depth is too large for this platform")


@_verify_stream_arg
def stream2sig(stream, depth):
    """
    Compute the signature of a stream
    """
    if depth <= 0:
        raise ValueError("Depth must be at least 1")
    elif depth == 1:
        return numpy.concatenate([[1.0], numpy.sum(numpy.diff(stream, axis=0), axis=0)])

    _verify_valid_depth(stream.shape[1], depth)

    backend = get_backend()
    return backend.compute_signature(stream, depth)


@_verify_stream_arg
def stream2logsig(stream, depth):
    """
    Compute the log signature of a stream
    """
    if depth <= 0:
        raise ValueError("Depth must be at least 1")
    elif depth == 1:
        return numpy.sum(numpy.diff(stream, axis=0), axis=0)

    _verify_valid_depth(stream.shape[1], depth)

    backend = get_backend()
    return backend.compute_log_signature(stream, depth)


def logsigdim(dimension, depth):
    """
    Get the number of elements in the log signature
    """
    if dimension == 0:
        raise ValueError("Dimension 0 is invalid")
    if depth == 1:
        return depth
    return get_backend().log_sig_dim(dimension, depth)


def sigdim(dimension, depth):
    """
    Get the number of elements in the signature
    """
    if dimension == 0:
        raise ValueError("Dimension 0 is invalid")
    if depth == 1:
        return dimension
    return get_backend().sig_dim(dimension, depth)


def logsigkeys(dimension, depth):
    """
    Get the keys that correspond to the elements in the log signature
    """
    return get_backend().log_sig_keys(dimension, depth)


def sigkeys(dimension, depth):
    """
    Get the keys that correspond to the elements in the signature
    """
    return get_backend().sig_keys(dimension, depth)
