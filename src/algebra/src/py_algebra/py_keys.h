//
// Created by user on 28/06/22.
//

#ifndef ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_KEYS_H_
#define ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_KEYS_H_
#include <pybind11/pybind11.h>

namespace esig {
namespace algebra {

void init_py_lie_key(pybind11::module_& m);
void init_py_tensor_key(pybind11::module_& m);

} // namespace algebra
} // namespace esig


#endif//ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_KEYS_H_
