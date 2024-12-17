import unittest
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import esig
import esig as ts

from . import auxiliaryfunct as ax
from . import esigtests as rado
from . import recombinetests as recombine

# # Tree-like equivalence of paths and equivalence class signature invariance. Non-generic paths used since pruning is generically unnecessary.
#
# a = [[0.0, 0], [1, 0], [1, 1], [1, 0], [2, 0], [3, 1], [2, 2], [1, 1], [2, 2], [1, 3], [2, 2], [3, 3], [2, 2], [3, 1],
#      [4, 1], [3, 1], [2, 0], [2, -1], [2, 0], [1, 0], [1, -1], [1, 0], [0, 0]]
#
# b = [[0.0, 0], [1, 3], [0, 0], [1, 5], [2, 5], [1, 5], [0, 6], [1, 5], [0, 0]]
#
# c = [[0.0, 0], [1, 1], [3, 1], [2, 1], [1, 1], [2, 0]]
#
# aa = ax.fullprune(a)
# bb = ax.fullprune(b)
# cc = ax.fullprune(c)
# a = np.array(a)
# b = np.array(b)
# c = np.array(c)
# aa = np.array(aa)
# bb = np.array(bb)
# cc = np.array(cc)
# sigdiffa = ts.stream2sig(a, 3) - ts.stream2sig(aa, 3)
# sigdiffb = ts.stream2sig(b, 3) - ts.stream2sig(bb, 3)
# sigdiffc = ts.stream2sig(c, 3) - ts.stream2sig(cc, 3)
#
# y = ax.random_path(1, range(-10, 11), 3)
#
# x = [[0.0, 0, 0]]
# xx = x
#
# xx.append(y[0])
# x = np.array(xx)
#
# path1 = ax.random_path(5, range(-10, 11), 3)
# path2 = ax.random_path(5, range(-10, 11), 3)
# # print(path1)
# # print(ax.concatenate(path1,path2))
#
# sig1 = ax.tensoraslevels(ts.stream2sig(np.array(path1), 2), 3, 2)
# # print(sig1)
# sig2 = ax.tensoraslevels(ts.stream2sig(np.array(path2), 2), 3, 2)
# # print(sig2)
# sig_concat = ax.tensoraslevels(ts.stream2sig(ax.concatenate(path1, path2), 2), 3, 2)
# chen = ax.chen(sig1, sig2)
# # print(chen)
# # print(sig_concat)
#
#
# logsig = ax.exponentiateTensor(ax.tensoraslevels(ax.Logsigastensor(np.array(path1), 3), 3, 3))
#

# c=ax.Logsigastensor(a,3)
# b=ts.stream2sig(a,3)
# d=ax.exponentiateTensor(ax.tensoraslevels(c,2,3))
# print(b)
# print(d)
# b=np.array(b)
# sigdiff= ts.stream2sig(a,2)-ts.stream2sig(b,2)
# print(sigdiff)

# if all(sigdiff)==0:
#   print("ok")


# Test Chen's theorem using tensor multiplication
# Compute esig of linear functions by hand and see it matches

# Exponentiate logsig as a tensor and see it matches esig


# a=[1,2]
# sig1=ax.linsig(a,3)
# b=[[0.0,0], [1,2]]
# b=np.array(b)
# sig2=ts.stream2sig(np.array(b),3)
# print(sig1-sig2)


# print(ax.fullunbracket(string))
# a=ax.Lietotensor(string)
# print(a)


# print(expand)

# L=ax.Lietotensor(string)
# print(L)
# print(ax.Lietotensor('1'))

# print(ax.changeofbasis(2,3))


# print(ax.Lietotensor(string))
# tensor_product=ax.fullunbracket(string)
# copy= tensor_product[1:-1]
# print(copy)

# term=ax.spotfirstterm(copy)
# print(type(term))
# print(term)
# d=ax.distribute(term)
# print(d)
# print(type(d))
##print(copy.replace(term,d))
# copy=copy.replace(term,ax.distribute(term))
# print(copy)
# term=ax.spotfirstterm(copy)
# print(term)
# print(type(term))
# a=ax.fullunbracket(string)

# print(a)
# print(len(a))
# b=a[1:-1]
# c=ax.spotterm(b)[1][1]
# print(c)

# term='1x2x(2x1-1x2)x3'
# print(ax.distribute(term))


class TestESIG(TestCase):

    def test_equivalence(self):
        a = [
            [0.0, 0],
            [1, 0],
            [1, 1],
            [1, 0],
            [2, 0],
            [3, 1],
            [2, 2],
            [1, 1],
            [2, 2],
            [1, 3],
            [2, 2],
            [3, 3],
            [2, 2],
            [3, 1],
            [4, 1],
            [3, 1],
            [2, 0],
            [2, -1],
            [2, 0],
            [1, 0],
            [1, -1],
            [1, 0],
            [0, 0],
        ]

        b = [[0.0, 0], [1, 3], [0, 0], [1, 5], [2, 5], [1, 5], [0, 6], [1, 5], [0, 0]]

        c = [[0.0, 0], [1, 1], [3, 1], [2, 1], [1, 1], [2, 0]]

        aa = np.array(ax.fullprune(a))
        bb = np.array(ax.fullprune(b))
        cc = np.array(ax.fullprune(c))
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        sigdiffa = ts.stream2sig(a, 3) - ts.stream2sig(aa, 3)
        sigdiffb = ts.stream2sig(b, 3) - ts.stream2sig(bb, 3)
        sigdiffc = ts.stream2sig(c, 3) - ts.stream2sig(cc, 3)

        self.assertEqual(all(sigdiffa), 0)
        self.assertEqual(all(sigdiffb), 0)
        self.assertEqual(all(sigdiffc), 0)

    def test_linsig(self):
        x = [[0.0, 0, 0]]
        xx = x

        y = ax.random_path(1, range(-10, 11), 3)
        xx.append(y[0])
        x = np.array(xx)

        # self.assertEqual(all(ax.linsig(y[0],4) - ts.stream2sig(x,4)),0)
        assert_array_almost_equal(ax.linsig(y[0], 4), ts.stream2sig(x, 4))

    def test_logsig(self):
        path1 = ax.random_path(5, range(-10, 11), 3)
        logsig = ax.exponentiateTensor(
            ax.tensoraslevels(ax.Logsigastensor(np.array(path1), 3), 3, 3)
        )

        # self.assertEqual(all(ts.stream2sig(np.array(path1),3)[1:]- logsig),0)
        assert_array_almost_equal(ts.stream2sig(np.array(path1), 3)[1:], logsig)

    def test_chen(self):
        path1 = ax.random_path(5, range(-10, 11), 3)
        path2 = ax.random_path(5, range(-10, 11), 3)

        sig1 = ax.tensoraslevels(ts.stream2sig(np.array(path1), 2), 3, 2)
        sig2 = ax.tensoraslevels(ts.stream2sig(np.array(path2), 2), 3, 2)

        sig_concat = ax.tensoraslevels(
            ts.stream2sig(ax.concatenate(path1, path2), 2), 3, 2
        )
        chen = ax.chen(sig1, sig2)
        # self.assertEqual(all(np.concatenate((np.array(chen)-np.array(sig_concat)).flatten())),0)
        for chen_sig, concat_sig in zip(chen, sig_concat):
            assert_array_equal(chen_sig, concat_sig)

    def test_compare_reverse(self):
        self.assertEqual(rado.compare_reverse_test(100, [-1, 0, 1], 2, 4), 1)

    def test_shuffle_test(self):
        self.assertEqual(rado.shuffle_test(100, [-1, 0, 1], 3, 3), 1)

    def test_reparametrizing(self):
        self.assertEqual(rado.reparametrizing_test(10, [-1, 0, 1], 2, 3, 3), 1)


# class TestRecombine(TestCase):
#
#     @unittest.skipIf(esig.NO_RECOMBINE, "Recombine not installed")
#     def test_recombine(self):
#         self.assertEqual(recombine.test(), 0)


class TestEsigFunctions(TestCase):

    def test_sig_from_transpose_data(self):
        # https://github.com/datasig-ac-uk/esig/issues/124
        two_dim_stream1 = np.array([[0.11212, 1.1212], [0.11, 0.22]])
        sig1 = esig.stream2sig(two_dim_stream1, 2)

        two_dim_stream2 = np.transpose(np.array([[0.11212, 0.11], [1.1212, 0.22]]))
        sig2 = esig.stream2sig(two_dim_stream2, 2)

        assert_array_equal(sig1, sig2)
