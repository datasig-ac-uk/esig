//
// Created by user on 08/12/22.
//

#include "py_scalars.h"

#include <pybind11/operators.h>

#include <esig/scalars.h>

#include "scalar_type.h"


using namespace esig;
using namespace esig::scalars;

static const char* SCALAR_DOC = R"edoc(
A generic scalar value.
)edoc";


void esig::python::init_scalars(py::module_ &m) {

    py::options options;
    options.disable_function_signatures();

    init_scalar_type(m);

    py::class_<scalar> klass(m, "Scalar", SCALAR_DOC);



}
