import math

import pytest
import numpy as np

try:
    from esig.paths import Stream, PiecewiseLiePath
    from esig.common import RealInterval
    from esig.algebra import Lie, FreeTensor, CoefficientType, VectorType, get_context
    skip = False
except ImportError:
    skip = True


WIDTH = 5
DEPTH = 3


@pytest.fixture(params=[1, 2, 5])
def count(request):
    return request.param


@pytest.fixture
def piecewise_intervals(count):
    return [RealInterval(float(i), float(i+1)) for i in range(count)]


@pytest.fixture
def piecewise_lie_data(piecewise_intervals, rng):
    return [
        (interval, Lie(rng.normal(0.0, 1.0, size=(WIDTH,)), width=WIDTH, depth=DEPTH))
        for interval in piecewise_intervals
    ]


@pytest.fixture
def piecewise_lie(piecewise_lie_data):
    return Stream(piecewise_lie_data, type=PiecewiseLiePath, width=WIDTH, depth=DEPTH)

@pytest.mark.skipif(skip, reason="path type not available")
def test_log_signature_full_data(piecewise_lie_data):
    ctx = get_context(WIDTH, DEPTH, CoefficientType.DPReal)
    piecewise_lie = Stream(piecewise_lie_data, type=PiecewiseLiePath, width=WIDTH, depth=DEPTH)

    result = piecewise_lie.log_signature(0.01)
    expected = ctx.cbh([d[1] for d in piecewise_lie_data], vec_type=VectorType.DenseVector)
    assert result == expected
