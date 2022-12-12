//
// Created by user on 08/12/22.
//

#include "py_esig.h"
//#include <esig/config.h>

#include "py_intervals.h"
#include "py_scalars.h"
#include "py_algebra.h"
#include "py_paths.h"
#include "recombine.h"

#ifndef ESIG_VERSION_STRING
#define ESIG_VERSION_STRING "1.0.0"
#endif

PYBIND11_MODULE(_esig, m) {

    m.add_object("__version__", py::str(ESIG_VERSION_STRING));

    esig::python::init_intervals(m);
    esig::python::init_scalars(m);
    esig::python::init_recombine(m);

    esig::python::init_algebra(m);
    esig::python::init_paths(m);




}
