//
// Created by sam on 12/12/22.
//

#ifndef ESIG_SRC_PYESIG_PY_TENSOR_KEY_ITERATOR_H_
#define ESIG_SRC_PYESIG_PY_TENSOR_KEY_ITERATOR_H_

#include "py_esig.h"

#include <limits>

#include "py_tensor_key.h"


namespace esig { namespace python {

class py_tensor_key_iterator {
    key_type m_current;
    key_type m_end;
    deg_t m_width, m_depth;

public:
    py_tensor_key_iterator(deg_t width, deg_t depth, key_type current = 0, key_type end = std::numeric_limits<key_type>::max());

    py_tensor_key next();
};

void init_tensor_key_iterator(pybind11::module_ &m);


}}

#endif//ESIG_SRC_PYESIG_PY_TENSOR_KEY_ITERATOR_H_
