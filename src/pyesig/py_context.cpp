//
// Created by user on 10/02/23.
//

#include "py_context.h"


#include <esig/algebra/context.h>
using namespace esig;
using namespace esig::algebra;

namespace py = pybind11;
using namespace py::literals;

void esig::python::init_context(py::module_& m) {

    py::class_<context> klass(m, "Context");

    klass.def_property_readonly("width", &context::width);
    klass.def_property_readonly("depth", &context::depth);
    klass.def_property_readonly("ctype", &context::ctype);
    klass.def("lie_size", &context::lie_size, "degree"_a);
    klass.def("tensor_size", &context::tensor_size, "degree"_a);
    klass.def("cbh", &context::cbh, "lies"_a, "vec_type"_a);




    m.def("get_context",
        [](deg_t width, deg_t depth, const py::capsule& ctype_c, std::vector<std::string> other) {
            return get_context(width, depth, ctype_c.get_pointer<const scalars::scalar_type>(), other);
        }, "width"_a, "depth"_a, "coeffs"_a, "other"_a);


}
