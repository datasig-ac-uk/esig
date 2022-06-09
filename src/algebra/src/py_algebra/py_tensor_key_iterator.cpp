//
// Created by user on 08/06/22.
//

#include "py_tensor_key_iterator.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace esig {
namespace algebra {

static const char* TKEY_ITERATOR_DOC = R"eadoc(Iterator over tensor words.
)eadoc";


py_tensor_key_iterator::py_tensor_key_iterator(const context *ctx, key_type current, key_type end)
    : p_ctx(ctx), m_current(current), m_end(end)
{
    auto max_size = p_ctx->tensor_size(p_ctx->depth());
    if (m_end > max_size) {
        m_end = max_size;
    }
}
py_tensor_key py_tensor_key_iterator::next()
{
    if (m_current >= m_end) {
        throw py::stop_iteration();
    }
    auto current = m_current;
    ++m_current;
    return py_tensor_key(p_ctx, current);
}


void init_tensor_key_iterator(pybind11::module_ &m)
{
    py::class_<py_tensor_key_iterator> klass(m, "TensorKeyIterator", TKEY_ITERATOR_DOC);

    klass.def(py::init([](const py_tensor_key& start_key) {
        return py_tensor_key_iterator(start_key.get_context(), static_cast<key_type>(start_key));
    }), "start_key"_a);
    klass.def(py::init([](const py_tensor_key& start_key, const py_tensor_key& end_key) {
        return py_tensor_key_iterator(start_key.get_context(), static_cast<key_type>(start_key), static_cast<key_type>(end_key));
    }), "start_key"_a, "end_key"_a);


    klass.def("__next__", &py_tensor_key_iterator::next);
}


}// namespace algebra
}// namespace esig
