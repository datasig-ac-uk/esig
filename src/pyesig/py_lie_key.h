//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_PY_LIE_KEY_H_
#define ESIG_SRC_PYESIG_PY_LIE_KEY_H_

#include "py_esig.h"

#include <boost/container/small_vector.hpp>

#include "py_lie_letter.h"

namespace esig { namespace python {

class py_lie_key {
public:
    using container_type = boost::container::small_vector<py_lie_letter, 2>;

private:
    container_type m_data;
    deg_t m_width;

public:
    explicit py_lie_key(deg_t width);
    explicit py_lie_key(deg_t width, let_t letter);
    explicit py_lie_key(deg_t width, const boost::container::small_vector_base<py_lie_letter> &data);
    explicit py_lie_key(deg_t width, let_t left, let_t right);
    explicit py_lie_key(deg_t width, let_t left, const py_lie_key &right);
    explicit py_lie_key(deg_t width, const py_lie_key &left, const py_lie_key &right);

    bool is_letter() const noexcept;
    let_t as_letter() const;
    std::string to_string() const;
    py_lie_key lparent() const;
    py_lie_key rparent() const;

    deg_t degree() const;

    bool equals(const py_lie_key &other) const noexcept;
};


void init_py_lie_key(py::module_& m);

}}

#endif//ESIG_SRC_PYESIG_PY_LIE_KEY_H_
