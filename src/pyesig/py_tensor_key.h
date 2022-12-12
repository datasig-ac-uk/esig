//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_PY_TENSOR_KEY_H_
#define ESIG_SRC_PYESIG_PY_TENSOR_KEY_H_

#include "py_esig.h"


namespace esig { namespace python {

class py_tensor_key {
    key_type m_key;
    deg_t m_width;
    deg_t m_depth;

public:
    explicit py_tensor_key(key_type key, deg_t width, deg_t depth);

    explicit operator key_type() const noexcept;

    std::string to_string() const;
    py_tensor_key lparent() const;
    py_tensor_key rparent() const;
    bool is_letter() const;

    deg_t width() const;
    deg_t depth() const;

    deg_t degree() const;
    std::vector<let_t> to_letters() const;

    bool equals(const py_tensor_key &other) const noexcept;
    bool less(const py_tensor_key &other) const noexcept;
};
}}


#endif//ESIG_SRC_PYESIG_PY_TENSOR_KEY_H_
