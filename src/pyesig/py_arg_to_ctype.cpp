//
// Created by user on 09/12/22.
//

#include "py_arg_to_ctype.h"


#include "scalar_meta.h"

const esig::scalars::scalar_type *esig::python::py_arg_to_ctype(const py::object &arg) {

    if (py::isinstance<py::str>(arg)) {
        return esig::scalars::get_type(arg.cast<std::string>());
    }

    return esig::python::to_stype_ptr(arg);
}
