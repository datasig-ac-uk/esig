import warnings
import numpy as np


warnings.warn("Importing functions from \"esig.tosig\" is deprecated and will be"
              " removed in version 1.2. Import the functions from the main \"esig\" "
              "package instead.", DeprecationWarning)


from esig.common import recombine


def stream2sig(stream, signature_degree):
    """
    stream2sig(array(no_of_ticks x signal_dimension),
    signature_degree) reads a 2 dimensional numpy array
    of floats, \"the data in stream space\" and returns
    a numpy vector containing the signature of the vector
    series up to given signature degree.
    """


def stream2logsig(stream, signature_degree):
    """
    stream2logsig(array(no_of_ticks x signal_dimension),
    signature_degree) reads a 2 dimensional numpy array
    of floats, \"the data in stream space\" and returns
    a numpy vector containing the log signature of the
    vector series up to given log signature degree
    """


def sigdim(signal_dimension, signature_degree):
    """
    sigdim(signal_dimension, signature_degree) returns
    an integer giving the length of the
    signature vector returned by stream2sig
    """


def logsigdim(signal_dimension, signature_degree):
    """
    logsigdim(signal_dimension, signature_degree) returns
    a Py_ssize_t integer giving the dimension of the log
    signature vector returned by stream2logsig
    """


def logsigkeys(signal_dimension, signature_degree):
    """
    logsigkeys(signal_dimension, signature_degree) returns,
    in the order used by stream2logsig, a space separated ascii
    string containing the keys associated the entries in the
    log signature returned by stream2logsig
    """


def sigkeys(signal_dimension, signature_degree):
    """
    sigkeys(signal_dimension, signature_degree) returns,
    in the order used by stream2sig, a space separated ascii
    string containing the keys associated the entries in
    the signature returned by stream2sig
    """
