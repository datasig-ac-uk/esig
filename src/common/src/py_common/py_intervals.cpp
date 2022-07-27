//
// Created by user on 04/05/22.
//

#include "py_intervals.h"
#include "py_common.h"
#include <esig/intervals.h>
#include <sstream>

namespace py = pybind11;
using namespace pybind11::literals;

static const char* INTERVAL_TYPE_DOC = R"edoc(The type of a half-open interval, either right open (clopen) or left open (opencl).
)edoc";

static const char* INTERVAL_DOC = R"edoc(Half-open interval in the real line.
)edoc";



void esig::init_intervals(py::module_& m)
{
    using esig::interval;
    using esig::interval_type;

    py::options options;
    options.disable_function_signatures();

    py::enum_<interval_type>(m, "IntervalType", INTERVAL_TYPE_DOC)
            .value("Clopen", interval_type::clopen)
            .value("Opencl", interval_type::opencl)
            .export_values();


    py::class_<interval, esig::py_interval> klass(m, "Interval", INTERVAL_DOC);

    klass.def_property_readonly("interval_type", &interval::get_type);

    klass.def("inf", &interval::inf);
    klass.def("sup", &interval::sup);
    klass.def("included_end", &interval::included_end);
    klass.def("excluded_end", &interval::excluded_end);
    klass.def("__eq__", [](const interval& lhs, const interval& rhs) { return lhs == rhs; });
    klass.def("__neq__", [](const interval& lhs, const interval& rhs) { return lhs != rhs; });
    klass.def("intersects_with", &interval::intersects_with, "other"_a);

    klass.def("contains",
              static_cast<bool (interval::*)(param_t) const noexcept>(&interval::contains),
              "arg"_a);
    klass.def("contains",
              static_cast<bool (interval::*)(const interval&) const noexcept>(&interval::contains),
              "arg"_a);

    klass.def("__repr__", [](const interval& arg) {
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
    klass.def("__str__", [](const interval& arg) {
        std::stringstream ss;
        ss << arg;
        return ss.str();
    });

    // init specific realizations of intervals.
    esig::init_real_intervals(m);
    esig::init_dyadic(m);
    esig::init_dyadic_interval(m);
}





esig::param_t esig::py_interval::inf() const
{
    PYBIND11_OVERRIDE_PURE(param_t, esig::interval, inf);
}
esig::param_t esig::py_interval::sup() const
{
    PYBIND11_OVERRIDE_PURE(param_t, esig::interval, sup);
}
esig::param_t esig::py_interval::excluded_end() const
{
    PYBIND11_OVERRIDE(param_t, esig::interval, excluded_end);
}
esig::param_t esig::py_interval::included_end() const
{
    PYBIND11_OVERRIDE(param_t, esig::interval, included_end);
}
bool esig::py_interval::contains(esig::param_t arg) const noexcept
{
    PYBIND11_OVERRIDE(bool, esig::interval, contains, arg);
}
bool esig::py_interval::is_associated(const esig::interval &arg) const noexcept
{
    PYBIND11_OVERRIDE(bool, esig::interval, is_associated, arg);
}
bool esig::py_interval::contains(const esig::interval &arg) const noexcept
{
    PYBIND11_OVERRIDE(bool, esig::interval, contains, arg);
}
