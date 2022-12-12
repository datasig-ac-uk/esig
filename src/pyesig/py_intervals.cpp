//
// Created by user on 08/12/22.
//

#include "py_intervals.h"

#include "dyadic.h"
#include "interval.h"
#include "dyadic_interval.h"
#include "real_interval.h"
#include "segmentation.h"

void esig::python::init_intervals(py::module_ &m) {

    init_dyadic(m);
    init_py_interval(m);
    init_dyadic_interval(m);
    init_real_interval(m);
    init_segmentation(m);

}
