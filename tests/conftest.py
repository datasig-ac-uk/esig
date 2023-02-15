
import pytest
from numpy.random import default_rng

from esig import _esig

@pytest.fixture(params=range(2, 6))
def width(request):
    return request.param


@pytest.fixture(params=range(2, 6))
def depth(request):
    return request.param


@pytest.fixture
def rng():
    return default_rng(12345)


@pytest.fixture
def tensor_size(width, depth):
    s = depth
    r = 1
    while s:
        r *= width
        r += 1
        s -= 1
    return r


@pytest.fixture(params=[_esig.DenseVector, _esig.SparseVector])
def vec_type(request):
    return request.param


@pytest.fixture(params=[_esig.double, _esig.float])
def coeff_type(request):
    return request.param
