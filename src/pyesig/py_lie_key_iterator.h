//
// Created by sam on 12/12/22.
//

#ifndef ESIG_SRC_PYESIG_PY_LIE_KEY_ITERATOR_H_
#define ESIG_SRC_PYESIG_PY_LIE_KEY_ITERATOR_H_

#include "py_esig.h"

#include <limits>

#include <esig/algebra/context.h>

#include "py_lie_key.h"


namespace esig { namespace python {

class py_lie_key_iterator {
    key_type m_current;
    key_type m_end;
    const algebra::context *p_ctx;

public:
    explicit py_lie_key_iterator(const algebra::context *ctx, key_type current = 1, key_type end = std::numeric_limits<key_type>::max());
    py_lie_key next();
};

void init_lie_key_iterator(py::module_ &m);
}}

#endif//ESIG_SRC_PYESIG_PY_LIE_KEY_ITERATOR_H_
