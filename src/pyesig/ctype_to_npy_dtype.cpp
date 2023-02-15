//
// Created by user on 09/12/22.
//
#ifndef ESIG_NO_NUMPY
#include "ctype_to_npy_dtype.h"

#include <esig/scalars.h>

py::dtype esig::python::ctype_to_npy_dtype(const esig::scalars::scalar_type *type) {
    if (type == scalars::dtl::scalar_type_holder<double>::get_type()) {
        return py::dtype("d");
    }
    if (type == scalars::dtl::scalar_type_holder<float>::get_type()) {
        return py::dtype("f");
    }

    throw py::type_error("unsupported data type");
}

#endif
