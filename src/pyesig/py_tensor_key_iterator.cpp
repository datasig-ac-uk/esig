//
// Created by sam on 12/12/22.
//

#include "py_tensor_key_iterator.h"

using namespace esig;
using namespace esig::python;
using namespace esig::python::maths;
using namespace pybind11::literals;


static const char *TKEY_ITERATOR_DOC = R"eadoc(Iterator over tensor words.
)eadoc";

py_tensor_key_iterator::py_tensor_key_iterator(deg_t width, deg_t depth, key_type current, key_type end)
    : m_current(current), m_end(end), m_width(width), m_depth(depth) {
    assert(width != 0 && width != 1);
    auto max_size = (power(dimn_t(width), depth + 1) - 1) / (dimn_t(width) - 1);
    if (m_end > max_size) {
        m_end = max_size;
    }
}
py_tensor_key py_tensor_key_iterator::next() {
    if (m_current >= m_end) {
        throw py::stop_iteration();
    }
    auto current = m_current;
    ++m_current;
    return py_tensor_key(current, m_width, m_depth);
}

void init_tensor_key_iterator(pybind11::module_ &m) {
    py::class_<py_tensor_key_iterator> klass(m, "TensorKeyIterator", TKEY_ITERATOR_DOC);

    klass.def(py::init([](const py_tensor_key &start_key) {
                  return py_tensor_key_iterator(start_key.width(), start_key.depth(), static_cast<key_type>(start_key));
              }),
              "start_key"_a);
    klass.def(py::init([](const py_tensor_key &start_key, const py_tensor_key &end_key) {
                  return py_tensor_key_iterator(start_key.width(), start_key.depth(), static_cast<key_type>(start_key), static_cast<key_type>(end_key));
              }),
              "start_key"_a, "end_key"_a);
    klass.def("__iter__", [](py_tensor_key_iterator &self) { return self; });
    klass.def("__next__", &py_tensor_key_iterator::next);
}
