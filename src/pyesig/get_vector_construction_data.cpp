//
// Created by user on 09/12/22.
//

#include "get_vector_construction_data.h"

#include "kwargs_to_vector_construction.h"

#include "py_scalars.h"

using namespace esig;
using namespace esig::algebra;
using namespace pybind11::literals;

vector_construction_data
esig::python::get_vector_construction_data(const py::object &data, const py::kwargs &kwargs, python::py_vector_construction_helper& helper) {

    python::py_to_buffer_options options;
    options.type = helper.ctype;

    auto data_buffer = python::py_to_buffer(data, options);

    return { std::move(data_buffer), helper.vtype };
}
