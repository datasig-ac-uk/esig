//
// Created by sam on 02/05/22.
//

#include "py_lie_increment_path.h"
#include "esig/paths/python_interface.h"

namespace py = pybind11;

void esig::paths::init_lie_increment_path(py::module_ &m)
{
    py::class_<esig::paths::lie_increment_path, esig::paths::path_interface> klass(m, "LieIncrementPath");

    esig::paths::register_pypath_constructor(py::type::of<esig::paths::lie_increment_path>(),
            construct_lie_increment_path);



}


esig::paths::path esig::paths::construct_lie_increment_path(const pybind11::args &args, const pybind11::kwargs &kwargs)
{
    return esig::paths::path();
}
