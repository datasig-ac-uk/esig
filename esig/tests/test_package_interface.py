
import unittest

import numpy as np
import numpy.testing as nptesting

import esig

STREAM = np.array([
    [1.0, 1.0],
    [3.0, 4.0],
    [5.0, 2.0],
    [8.0, 6.0]
])

SIGNATURE = np.array([
    1.0, 7.0, 5.0, 24.5, 19.0, 16.0, 12.5
])

LOG_SIGNATURE = np.array([
    7. , 5. , 1.5
])

class ArrayTestCase(unittest.TestCase):
    RTOL = 1e-3
    ATOL = 0.0

    def assertEqual(self, actual, expected, *args, **kwargs):
        if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
            nptesting.assert_equal(actual, expected)
        super().assertEqual(actual, expected, *args, **kwargs)

    def assert_allclose(self, actual, desired):
        """
        Custom assert method for Numpy arrays, built using
        the Numpy testing suite assertion methods.
        """
        nptesting.assert_allclose(
            actual, desired, rtol=self.RTOL, atol=self.ATOL
        )


class TestSignatureInterface(ArrayTestCase):

    def test_calculation_width_2_depth_0_error(self):
        width = 2
        depth = 0

        with self.assertRaises(ValueError):
            sig = esig.stream2sig(STREAM, depth)

    def test_calculation_width_2_depth_1(self):
        # Pure Python implementation 
        width = 2
        depth = 1

        sig = esig.stream2sig(STREAM, depth)

        self.assert_allclose(sig, SIGNATURE[:3])

    def test_calculation_width_2_depth_2(self):
        # tosig implementation

        width = 2
        depth = 2

        sig = esig.stream2sig(STREAM, depth)
        self.assert_allclose(sig, SIGNATURE)

    def test_calculation_width_2_depth_100_error(self):
        # Depth out of bounds for width 2
        width = 2
        depth = 100

        pattern = "Legitimate depth of 2<->\\d+ for records with width 2 exceeds limit"
        with self.assertRaisesRegex(RuntimeError, pattern):
            sig = esig.stream2sig(STREAM, depth)

    def test_dtype_validation_float64_passes(self):

        width = 2
        depth = 2

        stream = np.array(
            [[1.0, 2.0],
             [3.0, 4.0],
             [5.0, 6.0]],
            dtype=np.float64
        )

        sig = esig.stream2sig(stream, depth)
        self.assertGreater(sig.size, 0)

    def test_dtype_validation_float32_passes(self):

        width = 2
        depth = 2

        stream = np.array(
            [[1.0, 2.0],
             [3.0, 4.0],
             [5.0, 6.0]],
            dtype=np.float32
        )

        sig = esig.stream2sig(stream, depth)
        self.assertGreater(sig.size, 0)

    def test_dtype_validation_int64_fails(self):

        width = 2
        depth = 2

        stream = np.array(
            [[1, 2],
             [3, 4],
             [5, 6]],
            dtype=np.int64
        )

        with self.assertRaises(TypeError):
            sig = esig.stream2sig(stream, depth)
        #self.assertGreater(sig.size, 0)

    def test_computed_size_vs_size(self):

        width = 2

        for depth in range(2, 10):
            with self.subTest(depth=depth):
                sig = esig.stream2sig(STREAM, depth)
                size = esig.sigdim(width, depth)
                self.assertEqual(size, sig.size)
        
    def test_signature_keys(self):
        width = 2
        depth = 2

        keys = esig.sigkeys(width, depth)
        self.assertEqual(keys, " () (1) (2) (1,1) (1,2) (2,1) (2,2)")


class TestLogSignatureInterface(ArrayTestCase):

    def test_calculation_width_2_depth_0_error(self):
        width = 2
        depth = 0

        with self.assertRaises(ValueError):
            log_sig = esig.stream2logsig(STREAM, depth)

    def test_calculation_width_2_depth_1(self):
        # Pure Python implementation 
        width = 2
        depth = 1

        log_sig = esig.stream2logsig(STREAM, depth)

        self.assert_allclose(log_sig, LOG_SIGNATURE[:2])

    def test_calculation_width_2_depth_2(self):
        # tosig implementation

        width = 2
        depth = 2

        log_sig = esig.stream2logsig(STREAM, depth)
        self.assert_allclose(log_sig, LOG_SIGNATURE)

    def test_calculation_width_2_depth_100_error(self):
        # Depth out of bounds for width 2
        width = 2
        depth = 100

        pattern = "Legitimate depth of 2<->\\d+ for records with width 2 exceeds limit"
        with self.assertRaisesRegex(RuntimeError, pattern):
            log_sig = esig.stream2logsig(STREAM, depth)

    def test_dtype_validation_float64_passes(self):

        width = 2
        depth = 2

        stream = np.array(
            [[1.0, 2.0],
             [3.0, 4.0],
             [5.0, 6.0]],
            dtype=np.float64
        )

        log_sig = esig.stream2logsig(stream, depth)
        self.assertGreater(log_sig.size, 0)

    def test_dtype_validation_float32_passes(self):

        width = 2
        depth = 2

        stream = np.array(
            [[1.0, 2.0],
             [3.0, 4.0],
             [5.0, 6.0]],
            dtype=np.float32
        )

        log_sig = esig.stream2logsig(stream, depth)
        self.assertGreater(log_sig.size, 0)

    def test_dtype_validation_int64_fails(self):

        width = 2
        depth = 2

        stream = np.array(
            [[1, 2],
             [3, 4],
             [5, 6]],
            dtype=np.int64
        )

        with self.assertRaises(TypeError):
            log_sig = esig.stream2logsig(stream, depth)
        #self.assertGreater(sig.size, 0)

    def test_computed_size_vs_size(self):

        width = 2

        for depth in range(2, 10):
            with self.subTest(depth=depth):
                log_sig = esig.stream2logsig(STREAM, depth)
                size = esig.logsigdim(width, depth)
                self.assertEqual(size, log_sig.size)
        
    def test_signature_keys(self):
        width = 2
        depth = 2

        keys = esig.logsigkeys(width, depth)
        self.assertEqual(keys, " 1 2 [1,2]")