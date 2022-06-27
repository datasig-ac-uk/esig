//
// Created by user on 04/05/22.
//

#ifndef ESIG_SRC_COMMON_SRC_PY_COMMON_PY_COMMON_H_
#define ESIG_SRC_COMMON_SRC_PY_COMMON_PY_COMMON_H_

#include <pybind11/pybind11.h>
#include "py_recombine.h"

namespace esig {

void init_real_intervals(pybind11::module_& m);
void init_dyadic(pybind11::module_& m);
void init_dyadic_interval(pybind11::module_& m);
void init_intervals(pybind11::module_& m);



} // namespace esig



#endif//ESIG_SRC_COMMON_SRC_PY_COMMON_PY_COMMON_H_
