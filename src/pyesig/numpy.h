//
// Created by user on 20/02/23.
//

#ifndef ESIG_SRC_PYESIG_NUMPY_H_
#define ESIG_SRC_PYESIG_NUMPY_H_
#ifndef ESIG_NO_NUMPY

#include "py_esig.h"

#include <pybind11/numpy.h>

#include <esig/scalars.h>

namespace esig { namespace python {

const scalars::ScalarType * npy_dtype_to_ctype(py::dtype dtype);
py::dtype ctype_to_npy_dtype(const scalars::ScalarType *);

std::string npy_dtype_to_identifier(py::dtype dtype);


}}

#endif
#endif//ESIG_SRC_PYESIG_NUMPY_H_
