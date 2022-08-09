//
// Created by sam on 06/06/22.
//

#include "py_free_tensor.h"
#include "esig/algebra/python_interface.h"
#include "py_keys.h"

#include <cmath>

#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace esig {
namespace algebra {
py_tensor_key::py_tensor_key(key_type key, deg_t width, deg_t depth)
    : m_key(key), m_width(width), m_depth(depth)
{
}
bool py_tensor_key::is_letter() const {
    return false;
}

py_tensor_key::operator key_type() const noexcept
{
    return m_key;
}
deg_t py_tensor_key::width() const {
    return m_width;
}
deg_t py_tensor_key::depth() const {
    return m_depth;
}
std::string py_tensor_key::to_string() const
{
    return std::to_string(1);
}
py_tensor_key py_tensor_key::lparent() const
{
    return py_tensor_key(0, m_width, m_depth);
}
py_tensor_key py_tensor_key::rparent() const
{
    return py_tensor_key(0, m_width, m_depth);
}
deg_t py_tensor_key::degree() const
{
    return m_depth;
}
bool py_tensor_key::equals(const py_tensor_key &other) const noexcept
{
    return m_key == other.m_key;
}
bool py_tensor_key::less(const py_tensor_key &other) const noexcept
{
    return m_key < other.m_key;
}

} // namespace algebra
} // namespace esig



namespace {


esig::algebra::py_tensor_key construct_key(const py::args& args, const py::kwargs& kwargs)
{
    py::tuple py_letters = args;
    if (!args.empty() && py::isinstance<py::tuple>(args[0])) {
        py_letters = args[0].cast<py::tuple>();
    }

    auto letters = py_letters.cast<std::vector<esig::let_t>>();

    esig::deg_t width = 0, depth = esig::deg_t(letters.size());

    auto max_elt = std::max_element(letters.begin(), letters.end());
    if (kwargs.contains("width")) {
        width = kwargs["width"].cast<esig::deg_t>();
    }
    else if (!letters.empty()) {
        width = *max_elt;
    }

    if (!letters.empty() && *max_elt > width) {
        throw py::value_error("letter value exceeds alphabet size");
    }

    esig::key_type result = 0;
    auto wwidth = esig::dimn_t(width);
    for (auto letter : letters) {
        result *= wwidth;
        result += esig::key_type(letter);
    }

    return esig::algebra::py_tensor_key(result, width, letters.size());
}



} // namespace




void esig::algebra::init_py_tensor_key(pybind11::module_ &m)
{
    using esig::algebra::py_tensor_key;
    py::class_<py_tensor_key> klass(m, "TensorKey");

    klass.def("__str__", &py_tensor_key::to_string);
}
