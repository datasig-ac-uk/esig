//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_PY_LIE_LETTER_H_
#define ESIG_SRC_PYESIG_PY_LIE_LETTER_H_

#include "py_esig.h"

#include <ostream>

namespace esig { namespace python {

class py_lie_letter {
    dimn_t m_data = 0;

    constexpr explicit py_lie_letter(dimn_t raw) : m_data(raw) {}

public:
    py_lie_letter() = default;

    static constexpr py_lie_letter from_letter(let_t letter) {
        return py_lie_letter(1 + (dimn_t(letter) << 1));
    }

    static constexpr py_lie_letter from_offset(dimn_t offset) {
        return py_lie_letter(offset << 1);
    }

    constexpr bool is_offset() const noexcept {
        return (m_data & 1) == 0;
    }

    explicit operator let_t() const noexcept {
        return esig::let_t(m_data >> 1);
    }

    explicit constexpr operator dimn_t() const noexcept {
        return m_data >> 1;
    }

    inline friend std::ostream &operator<<(std::ostream &os, const py_lie_letter &let) {
        return os << let.m_data;
    }
};

}}


#endif//ESIG_SRC_PYESIG_PY_LIE_LETTER_H_
