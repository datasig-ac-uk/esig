//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_CTYPE_TO_NPY_DTYPE_H_
#define ESIG_SRC_PYESIG_CTYPE_TO_NPY_DTYPE_H_
#ifndef ESIG_NO_NUMPY
#include "py_esig.h"

#include <pybind11/numpy.h>

#include <esig/scalars.h>

namespace esig { namespace python {

py::dtype ctype_to_npy_dtype(const scalars::scalar_type* type);

}}

#endif
#endif//ESIG_SRC_PYESIG_CTYPE_TO_NPY_DTYPE_H_
