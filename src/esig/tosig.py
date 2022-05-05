from ._tosig import *

try:
    from esig.tosig import recombine

    NO_RECOMBINE = False
except ImportError:
    NO_RECOMBINE = True

    def recombine(*args, **kwargs):
        raise NotImplementedError
