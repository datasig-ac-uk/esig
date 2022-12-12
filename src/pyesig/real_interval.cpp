//
// Created by user on 09/12/22.
//

#include "real_interval.h"

#include <sstream>

#include <esig/intervals.h>

using namespace esig;
using namespace pybind11::literals;

static const char* REAL_INTERVAL_DOC = R"edoc(A half-open interval in the real line.
)edoc";

void esig::python::init_real_interval(py::module_ &m) {

    py::class_<esig::real_interval, esig::interval> klass(m, "RealInterval", REAL_INTERVAL_DOC);
    klass.def(py::init<>());
    klass.def(py::init<esig::interval_type>());
    klass.def(py::init<double, double>(), "inf"_a, "sup"_a);
    klass.def(py::init<double, double, esig::interval_type>(), "inf"_a, "sup"_a, "interval_type"_a);
    klass.def(py::init<const esig::interval&>(), "arg"_a);
    klass.def("__repr__", [](const real_interval& arg) {
        std::stringstream ss;
        ss << "RealInterval(inf="
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
}
