import warnings
import numpy as np


warnings.warn("Importing functions from \"esig.tosig\" is deprecated and will be"
              " removed in version 1.2. Import the functions from the main \"esig\" "
              "package instead.", DeprecationWarning)

import esig.algebra
from esig.common import recombine
from esig.algebra import get_context
from esig.paths import Stream, LieIncrementPath


def stream2sig(stream, signature_degree):
    """
    stream2sig(array(no_of_ticks x signal_dimension),
    signature_degree) reads a 2 dimensional numpy array
    of floats, \"the data in stream space\" and returns
    a numpy vector containing the signature of the vector
    series up to given signature degree.
    """
    path = Stream(np.diff(stream, axis=0), depth=signature_degree, type=LieIncrementPath)
    sig = path.signature(0.0, stream.shape[0]+1, 0.1)
    return np.array(sig)


def stream2logsig(stream, signature_degree):
    """
    stream2logsig(array(no_of_ticks x signal_dimension),
    signature_degree) reads a 2 dimensional numpy array
    of floats, \"the data in stream space\" and returns
    a numpy vector containing the log signature of the
    vector series up to given log signature degree
    """
    path = Stream(np.diff(stream, axis=0), depth=signature_degree, type=LieIncrementPath)
    m = stream.shape[0] + 1
    lsig = path.log_signature(0.0, float(m), 0.1)
    return np.array(lsig)


def sigdim(signal_dimension, signature_degree):
    """
    sigdim(signal_dimension, signature_degree) returns
    an integer giving the length of the
    signature vector returned by stream2sig
    """
    return get_context(signal_dimension, signature_degree, esig.algebra.DPReal).tensor_size(signature_degree)


def logsigdim(signal_dimension, signature_degree):
    """
    logsigdim(signal_dimension, signature_degree) returns
    a Py_ssize_t integer giving the dimension of the log
    signature vector returned by stream2logsig
    """
    return get_context(signal_dimension, signature_degree, esig.algebra.DPReal).lie_size(signature_degree)


def logsigkeys(signal_dimension, signature_degree):
    """
    logsigkeys(signal_dimension, signature_degree) returns,
    in the order used by stream2logsig, a space separated ascii
    string containing the keys associated the entries in the
    log signature returned by stream2logsig
    """
    ctx = get_context(signal_dimension, signature_degree, esig.algebra.DPReal)
    return " " + " ".join(map(lambda k: str(k).strip("()"), ctx.iterator_lie_keys()))


def sigkeys(signal_dimension, signature_degree):
    """
    sigkeys(signal_dimension, signature_degree) returns,
    in the order used by stream2sig, a space separated ascii
    string containing the keys associated the entries in
    the signature returned by stream2sig
    """
    ctx = get_context(signal_dimension, signature_degree, esig.algebra.DPReal)
    return " " + " ".join(map(str, ctx.iterate_tensor_keys()))
