# Mechanism for switching backends for computing signatures


import abc
import threading

import numpy as np
import roughpy as rp

# try:
# except ImportError:
#    # Error occurs during build sequence, since tosig does not exist
# tosig = None


try:
    import iisignature
except ImportError:
    iisignature = None


BACKENDS = {}

_BACKEND_LOCK = threading.RLock()
_BACKEND_DEFAULT = None
_BACKEND_CONTAINER = threading.local()


def get_backend():
    """
    Get the current global backend used for computing signatures.
    """
    global _BACKEND_CONTAINER
    if not hasattr(_BACKEND_CONTAINER, "context"):
        with _BACKEND_LOCK:
            _BACKEND_CONTAINER.context = _BACKEND_DEFAULT()
    return _BACKEND_CONTAINER.context


def set_backend(cls_or_name):
    """
    Change the backend used for computing signatures globally.
    """
    if isinstance(cls_or_name, str):
        if not cls_or_name in BACKENDS:
            raise ValueError("%s does not name a valid backend" % cls_or_name)
        _BACKEND_CONTAINER.context = BACKENDS[cls_or_name]()
    elif isinstance(cls_or_name, BackendBase):
        _BACKEND_CONTAINER.context = cls_or_name
    elif issubclass(cls_or_name, BackendBase):
        _BACKEND_CONTAINER.context = cls_or_name()
    else:
        raise TypeError("Backend must be a subclass of the BackendBase class or str")


def set_default_backend(cls):
    """
    Set the default backend across all threads.
    :param cls: Backend class to use.
    """
    with _BACKEND_LOCK:
        _BACKEND_DEFAULT = cls


def list_backends():
    """
    Get a list of all available backends that can be used for computing
    signatures.
    """
    return list(BACKENDS)


class BackendBase(abc.ABC):
    """
    Base class for signature/log signature computation backends.

    The required methods are `compute_signature` and `compute_log_signature`,
    which take data in the form of a Numpy array and return a Numpy array
    containing the flattened signature or log signature.
    """

    @abc.abstractmethod
    def compute_signature(self, stream, depth):
        """
        Compute the signature of the stream to required depth
        """

    @abc.abstractmethod
    def compute_log_signature(self, stream, depth):
        """
        Compute the log signature of the stream to required depth
        """

    def log_sig_dim(self, dimension, depth):
        """
        Get the number of elements in the log signature
        """
        context = rp.get_context(dimension, depth, rp.DPReal)
        return context.lie_size(depth)

    def sig_dim(self, dimension, depth):
        """
        Get the number of elements in the signature
        """
        context = rp.get_context(dimension, depth, rp.DPReal)
        return context.tensor_size(depth)

    @abc.abstractmethod
    def log_sig_keys(self, dimension, depth):
        """
        Get the keys that correspond to the elements in the log signature
        """

    @abc.abstractmethod
    def sig_keys(self, dimension, depth):
        """
        Get the keys that correspond to the elements in the signature
        """


class RoughPyBackend(BackendBase):

    def __repr__(self):
        return "RoughPyBackend"

    def prepare_stream(self, stream_data, depth):
        no_samples, width = stream_data.shape
        increments = np.diff(stream_data, axis=0)
        indices = np.arange(0.0, 1.0, 1.0 / (no_samples - 1))

        context = rp.get_context(width, depth, rp.DPReal)
        stream = rp.LieIncrementStream.from_increments(increments, indices=indices, ctx=context)

        return stream


    def empty_signature(self, width, depth):
        array = np.zeros(self.sig_dim(width, depth), dtype=np.float64)
        array[0] = 1.
        return array

    def empty_log_signature(self, width, depth):
        array = np.zeros(self.log_sig_dim(width, depth), dtype=np.float64)
        return array

    def compute_signature(self, stream, depth):
        no_samples, width = stream.shape
        if no_samples == 1:
            return self.empty_signature(width, depth)

        rpy_stream = self.prepare_stream(stream, depth)
        return np.array(rpy_stream.signature(rp.RealInterval(0.0, 1.0)), copy=True)

    def compute_log_signature(self, stream, depth):
        no_samples, width = stream.shape

        if no_samples == 1:
            return self.empty_log_signature(width, depth)

        rpy_stream = self.prepare_stream(stream, depth)
        return np.array(rpy_stream.log_signature(rp.RealInterval(0.0, 1.0)), copy=True)

    def log_sig_keys(self, dimension, depth):
        context = rp.get_context(dimension, depth, rp.DPReal)
        return " " + " ".join(map(str, iter(context.lie_basis)))

    def sig_keys(self, dimension, depth):
        context = rp.get_context(dimension, depth, rp.DPReal)
        return  " " + " ".join(map(str, iter(context.tensor_basis)))


BACKENDS["roughpy"] = RoughPyBackend

# For backwards compatibility
LibalgebraBackend = RoughPyBackend
BACKENDS["libalgebra"] = RoughPyBackend


if iisignature:

    class IISignatureBackend(BackendBase):

        def __init__(self):
            self._log_sig_prepare_cache = {}

        def __repr__(self):
            return "IISignatureBackend"

        def prepare(self, dimension, depth):
            if (dimension, depth) in self._log_sig_prepare_cache:
                return self._log_sig_prepare_cache[(dimension, depth)]

            s = iisignature.prepare(dimension, depth)
            self._log_sig_prepare_cache[(dimension, depth)] = s
            return s

        def compute_signature(self, stream, depth):
            return np.concatenate([[1.0], iisignature.sig(stream, depth)], axis=0)

        def compute_log_signature(self, stream, depth):
            _, dim = stream.shape
            s = self.prepare(dim, depth)
            return iisignature.logsig(stream, s)

        def log_sig_keys(self, dimension, depth):
            s = self.prepare(dimension, depth)
            return iisignature.basis(dimension, depth)

    BACKENDS["iisignature"] = IISignatureBackend


# set the default backend
_BACKEND_DEFAULT = RoughPyBackend
