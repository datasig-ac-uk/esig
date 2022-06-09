//
// Created by sam on 27/05/22.
//


#include "py_iterator.h"


namespace py = pybind11;
using namespace pybind11::literals;


static const char* ALG_ITER_DOC = R"eadoc()eadoc";



void esig::algebra::init_py_iterator(pybind11::module_ &m)
{
    py::class_<py_algebra_iterator> py_alg_iter(m, "AlgebraIterator", ALG_ITER_DOC);

    py_alg_iter.def("__next__", &py_algebra_iterator::next);
    py_alg_iter.def("__repr__", [](const py_algebra_iterator& arg) { return "AlgebraIterator"; });


}
