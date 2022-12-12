//
// Created by user on 07/06/22.
//

#include <esig/algebra/python_interface.h>
#include "py_free_tensor.h"


namespace py = pybind11;

namespace esig {
namespace algebra {

py_free_tensor_iterator::py_free_tensor_iterator(algebra_iterator it, algebra_iterator end)
    : m_it(std::move(it)), m_end(std::move(end))
{}
std::pair<py_tensor_key, scalars::scalar> py_free_tensor_iterator::next()
{
    if (m_it == m_end) {
        throw py::stop_iteration();
    }
//    std::pair<py_tensor_key, coefficient> rv{py_tensor_key(m_it.get_context(), m_it->key()), m_it->value()};
//    ++m_it;
//    return rv;
    throw py::stop_iteration();
}

} // namespace algebra
} // namespace esig
