//
// Created by user on 08/06/22.
//

#include "py_lie_key_iterator.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace esig {
namespace algebra {


static const char* LKEY_ITERATOR_DOC = R"eadoc(Iterator over range of Hall set members.
)eadoc";



py_lie_key_iterator::py_lie_key_iterator(const context *ctx, key_type current, key_type end)
    : p_ctx(ctx), m_current(current), m_end(end)
{
    if (current == 0) {
        throw std::invalid_argument("Invalid starting Lie key");
    }
    auto max_key = p_ctx->lie_size(p_ctx->depth());
    if (m_end > max_key) {
        m_end = max_key;
    }

}
py_lie_key py_lie_key_iterator::next()
{
    if (m_current > m_end) {
        throw py::stop_iteration();
    }
    auto current = m_current;
    ++m_current;
    return py_lie_key(p_ctx, current);
}





void init_lie_key_iterator(pybind11::module_ &m)
{
    py::class_<py_lie_key_iterator> klass(m, "LieKeyIterator", LKEY_ITERATOR_DOC);
//    klass.def(py::init<const context*>(), "context"_a);
//    klass.def(py::init<const context*, key_type>(), "context"_a, "start_key"_a);
//    klass.def(py::init<const context*, key_type, key_type>(), "context"_a, "start_key"_a, "end_key"_a);

    klass.def(py::init([](const py_lie_key& start_key) {
             return py_lie_key_iterator(start_key.get_context(), static_cast<key_type>(start_key));
         }), "start_key"_a);
    klass.def(py::init([](const py_lie_key& start_key, const py_lie_key& end_key) {
             return py_lie_key_iterator(start_key.get_context(), static_cast<key_type>(start_key), static_cast<key_type>(end_key));
         }), "start_key"_a, "end_key"_a);

//    klass.def(py::init([](deg_t width, deg_t depth, coefficient_type ctype) {
//             auto ctx = get_context(width, depth, ctype);
//             return py_lie_key_iterator(ctx.get());
//         }), py::kw_only(), "width"_a, "depth"_a, "ctype"_a=coefficient_type::dp_real);
//    klass.def(py::init([](key_type start_key, deg_t width, deg_t depth, coefficient_type ctype) {
//        auto ctx = get_context(width, depth, ctype);
//        return py_lie_key_iterator(ctx.get(), start_key);
//    }), "start_key"_a, py::kw_only(), "width"_a, "depth"_a, "ctype"_a=coefficient_type::dp_real);


    klass.def("__iter__", [](py_lie_key_iterator& self) { return self; });
    klass.def("__next__", &py_lie_key_iterator::next);



}

}// namespace algebra
}// namespace esig
