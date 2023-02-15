//
// Created by user on 09/12/22.
//

#include "get_vector_construction_data.h"

#include "ctype_to_py_fmt.h"
#include "py_buffer_to_buffer.h"
#include "py_arg_to_ctype.h"
#include "kwargs_to_vector_construction.h"


using namespace esig;
using namespace esig::algebra;
using namespace pybind11::literals;

vector_construction_data
esig::python::get_vector_construction_data(const py::object &data, const py::kwargs &kwargs) {

    auto helper = python::kwargs_to_construction_data(kwargs);

    scalars::key_scalar_array data_buffer;

    if (py::isinstance<py::buffer>(data)) {
        auto info = py::reinterpret_borrow<py::buffer>(data).request();

        


    }



}
