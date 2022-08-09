//
// Created by user on 08/06/22.
//

#ifndef ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_TENSOR_KEY_ITERATOR_H_
#define ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_TENSOR_KEY_ITERATOR_H_

#include <esig/algebra/python_interface.h>
#include <esig/algebra/context.h>
#include <limits>

namespace esig {
namespace algebra {

class py_tensor_key_iterator
{
    key_type m_current;
    key_type m_end;
    deg_t m_width, m_depth;

public:

    py_tensor_key_iterator(deg_t width, deg_t depth, key_type current=0, key_type end=std::numeric_limits<key_type>::max());

    py_tensor_key next();

};


void init_tensor_key_iterator(pybind11::module_& m);

}// namespace algebra
}// namespace esig

#endif//ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_TENSOR_KEY_ITERATOR_H_
