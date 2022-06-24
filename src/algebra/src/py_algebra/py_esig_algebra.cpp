//
// Created by user on 18/03/2022.
//
//#include <pybind11/pybind11.h>

#include <esig/algebra/context.h>

#include "py_free_tensor.h"
#include "py_lie.h"
#include "py_coefficients.h"
#include "py_lie_key_iterator.h"
#include "py_tensor_key_iterator.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace esig;
using namespace esig::algebra;
using namespace pybind11::literals;

static const char* ALGEBRA_MOD_DOC = R"eadoc(Esig algebra module.

This module contains types that represent various algebraic objects including
elements of the (truncated) tensor algebra and free lie algebra.
)eadoc";



PYBIND11_MODULE(algebra, m)
{
    py::options options;
    options.disable_function_signatures();

    m.doc() = ALGEBRA_MOD_DOC;

    py::enum_<vector_type>(m, "VectorType")
            .value("DenseVector", vector_type::dense)
            .value("SparseVector", vector_type::sparse)
            .export_values();
    py::enum_<coefficient_type>(m, "CoefficientType")
            .value("DPReal", coefficient_type::dp_real)
            .value("SPReal", coefficient_type::sp_real)
            .export_values();

    init_py_coefficients(m);
    init_tensor_key_iterator(m);
    init_lie_key_iterator(m);

    py::class_<context, std::shared_ptr<context>> py_ctx(m, "Context");
    py_ctx.doc() = "Helper class for performing computations";

    py_ctx.def_property_readonly("width", &context::width);
    py_ctx.def_property_readonly("depth", &context::depth);
    py_ctx.def("lie_size", &context::lie_size, "degree"_a);
    py_ctx.def("tensor_size", &context::tensor_size, "degree"_a);
    py_ctx.def("cbh", &context::cbh, "lies"_a, "vec_type"_a);
    py_ctx.def("iterate_tensor_keys", [](const context& ctx) {
              return esig::algebra::py_tensor_key_iterator(&ctx);
          });
    py_ctx.def("iterator_lie_keys", [](const context& ctx) {
              return esig::algebra::py_lie_key_iterator(&ctx);
          });

    m.def("get_context", &get_context, "width"_a, "depth"_a, "ctype"_a, "preferences"_a);




    init_free_tensor(m);
    init_py_lie(m);
}
