//
// Created by user on 26/05/22.
//
#ifndef ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_COEFFICIENTS_H_
#define ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_COEFFICIENTS_H_

#include <esig/implementation_types.h>
#include <esig/algebra/coefficients.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace esig {
namespace algebra {

void init_py_coefficients(pybind11::module_& m);

namespace dtl {



} // namespace dtl



} // namespace algebra
} // namespace esig


#endif//ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_COEFFICIENTS_H_
