//
// Created by user on 08/12/22.
//

#include "py_intervals.h"

#include "dyadic.h"
#include "interval.h"
#include "dyadic_interval.h"
#include "real_interval.h"
#include "segmentation.h"

static const char* INTERVAL_TYPE_DOC = R"edoc(
The type of a half-open interval, either right open (clopen) or left open (opencl).
)edoc";

void esig::python::init_intervals(py::module_ &m) {

    py::enum_<esig::interval_type>(m, "IntervalType", INTERVAL_TYPE_DOC)
        .value("Clopen", interval_type::clopen)
        .value("Opencl", interval_type::opencl)
        .export_values();

    init_dyadic(m);
    init_py_interval(m);
    init_dyadic_interval(m);
    init_real_interval(m);
    init_segmentation(m);

}
