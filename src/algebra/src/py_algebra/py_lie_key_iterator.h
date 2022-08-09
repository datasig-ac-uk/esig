//
// Created by user on 08/06/22.
//

#ifndef ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_LIE_KEY_ITERATOR_H_
#define ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_LIE_KEY_ITERATOR_H_

#include <esig/algebra/python_interface.h>
#include <limits>

namespace esig {
namespace algebra {

class py_lie_key_iterator
{
    key_type m_current;
    key_type m_end;
    const context* p_ctx;

public:

    explicit py_lie_key_iterator(const context* ctx, key_type current=1, key_type end=std::numeric_limits<key_type>::max());
    py_lie_key next();

};


void init_lie_key_iterator(pybind11::module_& m);

}// namespace algebra
}// namespace esig

#endif//ESIG_SRC_ALGEBRA_SRC_PY_ALGEBRA_PY_LIE_KEY_ITERATOR_H_
