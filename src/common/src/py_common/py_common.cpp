//
// Created by user on 26/04/22.
//

#include <pybind11/pybind11.h>

#include "esig/config.h"
#include "esig/implementation_types.h"

#include "../segmentation.h"
#include "../tosig.h"
#include "py_common.h"

namespace py = pybind11;
using namespace pybind11::literals;


void setup_tosig(py::module_& m);


PYBIND11_MODULE(common, m) {
    m.add_object("__version__", py::str(ESIG_VERSION_STRING));

    esig::init_intervals(m);


    auto tosig_module = m.def_submodule("_tosig");
    setup_tosig(tosig_module);


    m.def("segment", &esig::segment_full, "interval"_a, "indicator_func"_a, "trim_tol"_a, "signal_tol"_a);
    m.def("segment", &esig::segment_simple, "interval"_a, "indicator_func"_a, "precision"_a);

}


void setup_tosig(py::module_ &m)
{
}
