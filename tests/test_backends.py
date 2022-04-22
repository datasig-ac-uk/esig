
import itertools
import unittest


import esig
from esig.backends import BackendBase




class TBackend(BackendBase):

    def compute_signature(self, stream, depth):
        pass

    def compute_log_signature(self, stream, depth):
        pass

    def log_sig_keys(self, dimension, depth):
        pass

    def sig_keys(self, dimension, depth):
        pass


class TestBackendDefaultMethods(unittest.TestCase):

    def setUp(self):
        self.backend = TBackend()
        self.sig_dim_func = lambda w, d: (w**(d+1) - 1)/(w - 1)

    def test_log_sig_dim_calc(self):
        width = 2
        depth = 2
        self.assertEqual(
            self.backend.log_sig_dim(width, depth),
            3
        )

    def test_sig_dim_calc(self):
        width = 2
        depth = 2
        self.assertEqual(
            self.backend.sig_dim(width, depth),
            self.sig_dim_func(width, depth)
        )


class TestBackendGetterMethods(unittest.TestCase):

    def setUp(self):
        esig.backends.BACKENDS["test"] = TBackend

    def tearDown(self):
        esig.set_backend(esig.backends.LibalgebraBackend)

    def test_get_default_backend(self):
        backend = esig.get_backend()

        self.assertTrue(isinstance(
            backend, 
            esig.backends.LibalgebraBackend
        ))

    def test_set_backend_by_class(self):
        backend = esig.get_backend()
        assert not isinstance(backend, TBackend)

        esig.set_backend(TBackend)
        self.assertTrue(isinstance(
            esig.get_backend(), 
            TBackend
        ))

    def test_set_backend_by_string(self):
        backend = esig.get_backend()
        assert not isinstance(backend, TBackend)

        esig.set_backend("test")
        self.assertTrue(isinstance(
            esig.get_backend(), 
            TBackend
        ))
