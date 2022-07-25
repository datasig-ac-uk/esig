
import itertools

import numpy as np
import pytest

from esig.algebra import FreeTensor, TensorKey


@pytest.fixture
def TensorKey_iter(width, depth):

    def itr():
        yield TensorKey(width=width, depth=depth)

        for let in range(1, width+1):
            yield TensorKey(let, width=width, depth=depth)

        for d in range(2, depth+1):
            for data in itertools.product(range(1, width+1), repeat=d):
                yield TensorKey(data, width=width, depth=depth)

    return itr


def test_FreeTensor_iterator(width, depth, FreeTensor_size, TensorKey_iter):
    data = np.arange(1.0, float(FreeTensor_size+1))
    tens = FreeTensor(data, width=width, depth=depth)

    result = list(tens)
    expected = list(zip(TensorKey_iter(), data))

    assert result == expected
