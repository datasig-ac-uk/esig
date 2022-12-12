//
// Created by sam on 03/05/22.
//

#ifndef ESIG_PY_FREE_TENSOR_H
#define ESIG_PY_FREE_TENSOR_H

#include <esig/algebra/basis.h>
#include <esig/algebra/context.h>
#include <esig/algebra/python_interface.h>
#include <esig/algebra/tensor_interface.h>
#include <pybind11/pybind11.h>

namespace esig {
namespace algebra {

class py_free_tensor_iterator
{
    algebra_iterator m_it;
    algebra_iterator m_end;

public:
    py_free_tensor_iterator(algebra_iterator it, algebra_iterator end);
    std::pair<py_tensor_key, scalars::scalar> next();
};



void init_free_tensor(pybind11::module_& m);

} // namespace algebra
} // namespace esig


#endif//ESIG_PY_FREE_TENSOR_H
