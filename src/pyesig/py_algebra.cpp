//
// Created by user on 08/12/22.
//

#include "py_algebra.h"

#include "free_tensor.h"
#include "lie.h"


void esig::python::init_algebra(py::module_ &m) {

    init_lie(m);
    init_free_tensor(m);

}
