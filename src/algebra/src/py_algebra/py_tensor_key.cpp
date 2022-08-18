//
// Created by sam on 06/06/22.
//

#include "py_free_tensor.h"
#include "esig/algebra/python_interface.h"
#include "py_keys.h"

#include <cmath>
#include <sstream>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;


namespace {

template <typename I, typename E>
constexpr I power(I arg, E exponent) noexcept
{
    if (exponent == 0) {
        return I(1);
    }
    if (exponent == 1) {
        return arg;
    }
    auto recurse = power(arg, exponent / 2);
    return recurse*recurse*(exponent & 1 == 1 ? arg : I(1));
}

}

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
std::vector<let_t> py_tensor_key::to_letters() const {
    std::vector<let_t> letters;
    letters.reserve(m_depth);
    auto tmp = m_key;
    while (tmp) {
        tmp -= 1;
        letters.push_back(1+ (tmp % m_width));
        tmp /= m_width;
    }
    std::reverse(letters.begin(), letters.end());
    return letters;
}
std::string py_tensor_key::to_string() const
{
    std::stringstream ss;
    ss << '(';
    bool not_first = false;
    for (auto letter : to_letters()) {
        if (not_first) {
            ss << ',';
        }
        ss << letter;
        not_first = true;
    }
    ss << ')';
    return ss.str();
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

    return m_width == other.m_width && m_key == other.m_key;
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
    using esig::let_t;
    using esig::deg_t;
    using esig::dimn_t;
    using esig::key_type;
    std::vector<let_t> letters;


    if (args.empty() && kwargs.contains("index")) {
        auto width = kwargs["width"].cast<deg_t>();
        auto depth = kwargs["depth"].cast<deg_t>();
        auto index = kwargs["index"].cast<key_type>();

        auto max_idx = (power(dimn_t(width), depth+1) - 1) / (dimn_t(width) - 1);
        if (index >= max_idx) {
            throw py::value_error("provided index exceeds maximum");
        }

        return esig::algebra::py_tensor_key(index, width, depth);
    }

    if (!args.empty() && py::isinstance<py::sequence>(args[0])) {
        letters.reserve(py::len(args[0]));
        for (auto arg : args[0]) {
            letters.push_back(arg.cast<let_t>());
        }
    } else {
        letters.reserve(py::len(args));
        for (auto arg : args) {
            letters.push_back(arg.cast<let_t>());
        }
    }

    esig::deg_t width = 0;
    esig::deg_t depth = esig::deg_t(letters.size());

    auto max_elt = std::max_element(letters.begin(), letters.end());
    if (kwargs.contains("width")) {
        width = kwargs["width"].cast<esig::deg_t>();
    }
    else if (!letters.empty()) {
        width = *max_elt;
    }

    if (kwargs.contains("depth")) {
        depth = kwargs["depth"].cast<esig::deg_t>();
    }

    if (letters.size() > depth) {
        throw py::value_error("number of letters exceeds specified depth");
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



    return esig::algebra::py_tensor_key(result, width, depth);
}



} // namespace




void esig::algebra::init_py_tensor_key(pybind11::module_ &m)
{
    using esig::algebra::py_tensor_key;
    py::class_<py_tensor_key> klass(m, "TensorKey");
    klass.def(py::init(&construct_key));


    klass.def_property_readonly("width", &py_tensor_key::width);
    klass.def_property_readonly("max_degree", &py_tensor_key::depth);

    klass.def("degree", [](const py_tensor_key& key) { return key.to_letters().size(); });

    klass.def("__str__", &py_tensor_key::to_string);
    klass.def("__repr__", &py_tensor_key::to_string);

    klass.def("__eq__", &py_tensor_key::equals);
}
