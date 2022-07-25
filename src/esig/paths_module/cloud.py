
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from esig.common import Interval, RealInterval
from esig.algebra import get_context, Lie, FreeTensor
from esig._paths import Stream, LieIncrementPath


class StreamCloud:
    """
    A collection of streams over a common domain.

    """

    def __init__(self, data=None, width=None, default_depth=None, **path_kwargs):
        self.default_depth = default_depth or 2
        if isinstance(data, np.ndarray):
            assert data.ndim == 3
            self.width = width or data.shape[1]
            self.num_streams = data.shape[2]
            self.streams = [Stream(data[:, :, i], depth=self.default_depth, width=width, type=LieIncrementPath, **path_kwargs)
                            for i in range(self.num_streams)]

        else:
            raise ValueError("No constructor available for this data type")

    @classmethod
    def _get_interval(cls, itvl, upper):
        if not isinstance(itvl, Interval):
            assert upper is not None
            itvl = RealInterval(lower=itvl, upper=upper)
        return itvl

    def signature(self, interval, upper=None, *, depth=None, njobs=None):
        interval = self._get_interval(interval, upper)
        depth = depth or self.default_depth

        return [stream.signature(interval, depth) for stream in self.streams]

    def log_signature(self, interval, upper=None, *, depth=None, **kwargs):
        interval = self._get_interval(interval, upper)
        depth = depth or self.default_depth

        return [stream.log_signature(interval, depth) for stream in self.streams]

    def aggregate_signature(self, function, interval, upper=None, *, depth=None):
        interval = self._get_interval(interval, upper)
        assert callable(function)

        return function(stream.signature(interval, depth=depth) for stream in self.streams)

    def aggregate_log_signature(self, function, interval, upper=None, *, depth=None):
        interval = self._get_interval(interval, upper)
        assert callable(function)

        return function(stream.log_signature(interval, depth=depth) for stream in self.streams)

    def mean_signature(self, interval, upper=None, *, depth=None):
        depth = depth or self.default_depth
        interval = self._get_interval(interval, upper)
        zero_tensor = FreeTensor(width=self.width, depth=depth)

        if self.num_streams == 0:
            return zero_tensor

        streams_iter = (stream.signature(interval, depth=depth) for stream in self.streams)
        return sum(streams_iter, zero_tensor) / self.num_streams

    def mean_log_signature(self, interval, upper=None, *, depth=None):
        depth = depth or self.default_depth
        interval = self._get_interval(interval, upper)
        zero_lie = Lie(width=self.width, depth=depth)

        if self.num_streams == 0:
            return zero_lie

        streams_iter = (stream.log_signature(interval, depth=depth) for stream in self.streams)
        return sum(streams_iter, zero_lie) / self.num_streams
