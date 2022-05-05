import warnings

from ._tosig import *

try:
    from esig.tosig import recombine

    NO_RECOMBINE = False
except ImportError:
    NO_RECOMBINE = True

    def recombine(*args, **kwargs):
        raise NotImplementedError

warnings.warn("Importing functions from \"esig.tosig\" is deprecated and will be"
              " removed in version 1.0. Import the functions from the main \"esig\" "
              "package instead.", DeprecationWarning)
