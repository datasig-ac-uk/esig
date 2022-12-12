//
// Created by user on 09/12/22.
//

#include "interval.h"

#include <sstream>

#include <pybind11/pybind11.h>



using namespace esig;
using namespace pybind11::literals;

esig::param_t esig::python::py_interval::inf() const {
    PYBIND11_OVERRIDE_PURE(param_t, esig::interval, inf);
}
esig::param_t esig::python::py_interval::sup() const {
    PYBIND11_OVERRIDE_PURE(param_t, esig::interval, sup);
}
esig::param_t esig::python::py_interval::included_end() const {
    PYBIND11_OVERRIDE(param_t, esig::interval, excluded_end);
}
esig::param_t esig::python::py_interval::excluded_end() const {
    PYBIND11_OVERRIDE(param_t, esig::interval, included_end);
}
bool esig::python::py_interval::contains(esig::param_t arg) const noexcept {
    PYBIND11_OVERRIDE(bool, esig::interval, contains, arg);
}
bool esig::python::py_interval::is_associated(const esig::interval &arg) const noexcept {
    PYBIND11_OVERRIDE(bool, esig::interval, is_associated, arg);
}
bool esig::python::py_interval::contains(const esig::interval &arg) const noexcept {
    PYBIND11_OVERRIDE(bool, esig::interval, contains, arg);
}


static const char* INTERVAL_DOC = R"edoc(
Half-open interval in the real line.
)edoc";


void python::init_py_interval(py::module_ &m) {

    py::options options;
    options.disable_function_signatures();

    py::class_<interval, esig::python::py_interval> klass(m, "Interval", INTERVAL_DOC);

    klass.def_property_readonly("interval_type", &interval::get_type);

    klass.def("inf", &interval::inf);
    klass.def("sup", &interval::sup);
    klass.def("included_end", &interval::included_end);
    klass.def("excluded_end", &interval::excluded_end);
    klass.def("__eq__", [](const interval &lhs, const interval &rhs) { return lhs == rhs; });
    klass.def("__neq__", [](const interval &lhs, const interval &rhs) { return lhs != rhs; });
    klass.def("intersects_with", &interval::intersects_with, "other"_a);

    klass.def("contains",
              static_cast<bool (interval::*)(param_t) const noexcept>(&interval::contains),
              "arg"_a);
    klass.def("contains",
              static_cast<bool (interval::*)(const interval &) const noexcept>(&interval::contains),
              "arg"_a);

    klass.def("__repr__", [](const interval &arg) {
        std::stringstream ss;
        ss << "Interval(inf="
           << std::to_string(arg.inf())
           << ", sup="
           << std::to_string(arg.sup())
           << ", type=";
        if (arg.get_type() == interval_type::clopen) {
            ss << "clopen";
        } else {
            ss << "opencl";
        }
        ss << ')';
        return ss.str();
    });
    klass.def("__str__", [](const interval &arg) {
        std::stringstream ss;
        ss << arg;
        return ss.str();
    });
}
