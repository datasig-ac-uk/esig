//
// Created by user on 09/12/22.
//

#include "py_arg_to_ctype.h"


#include "scalar_meta.h"

const esig::scalars::scalar_type *esig::python::py_arg_to_ctype(const py::object &arg) {
    return esig::python::to_stype_ptr(arg);
}
