//
// Created by user on 04/05/22.
//
#include "py_common.h"
#include <esig/intervals.h>

namespace py = pybind11;
using namespace pybind11::literals;

static const char* REAL_INTERVAL_DOC = R"edoc(A half-open interval in the real line.
)edoc";


void esig::init_real_intervals(pybind11::module_& m)
{
    py::class_<esig::real_interval, esig::interval> klass(m, "RealInterval", REAL_INTERVAL_DOC);
    klass.def(py::init<>());
    klass.def(py::init<esig::interval_type>());
    klass.def(py::init<double, double>(), "inf"_a, "sup"_a);
    klass.def(py::init<double, double, esig::interval_type>(), "inf"_a, "sup"_a, "interval_type"_a);
    klass.def(py::init<const esig::interval&>(), "arg"_a);
}
