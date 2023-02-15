import pytest
import numpy as np

from esig import RealInterval, FreeTensor
from esig.paths import StreamCloud


@pytest.fixture(params=[5, 10, 100])
def length(request):
    return request.param


@pytest.fixture(params=[1, 2, 5])
def npaths(request):
    return request.param


@pytest.fixture
def cloud_data(rng, width, length, npaths):
    return rng.uniform(-2.0, 2.0, size=(length, width, npaths))


@pytest.fixture
def indices(length):
    return np.linspace(0.0, 1.0, length)


def test_creation_multiarray(width, npaths, cloud_data, indices):
    cloud = StreamCloud(data=cloud_data, default_depth=2, indices=indices)

    assert cloud.num_streams == npaths
    assert len(cloud.streams) == npaths
    assert all(stream.width == width for stream in cloud.streams)


def test_cloud_signature(width, depth, cloud_data, indices):
    cloud = StreamCloud(data=cloud_data, indices=indices)
    interval = RealInterval(0.0, 1.0)

    signatures = cloud.signature(interval)
    assert len(signatures) == cloud.num_streams
    assert all(isinstance(sig, FreeTensor) for sig in signatures)
