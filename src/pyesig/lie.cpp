//
// Created by user on 09/12/22.
//

#include "lie.h"

#include <pybind11/operators.h>


#include <esig/scalars.h>
#include <esig/algebra/lie_interface.h>

#include "numpy.h"
#include "get_vector_construction_data.h"
#include "py_lie_key.h"
#include "py_scalars.h"

using namespace esig;
using namespace esig::algebra;
using namespace esig::python;
using namespace pybind11::literals;


static const char* LIE_DOC = R"edoc(
Element of the free Lie algebra.
)edoc";


static lie construct_lie(py::object data, py::kwargs kwargs) {
    auto helper = esig::python::kwargs_to_construction_data(kwargs);

    py_to_buffer_options options;
    options.type = helper.ctype;
    options.allow_scalar = false;



    auto buffer = py_to_buffer(data, options);


    if (helper.ctype == nullptr) {
        if (options.type == nullptr) {
            throw py::value_error("could not deduce an appropriate scalar_type");
        }
        helper.ctype = options.type;
    }

    if (helper.width == 0 && buffer.size() > 0) {
        helper.width = static_cast<deg_t>(buffer.size());
    }

    if (!helper.ctx) {
        if (helper.width == 0 || helper.depth == 0) {
            throw py::value_error("you must provide either context or both width and depth");
        }
        helper.ctx = get_context(helper.width, helper.depth, helper.ctype, {});
    }

    auto result = helper.ctx->construct_lie({std::move(buffer), helper.vtype});

    if (options.cleanup) {
        options.cleanup();
    }

    return result;
}



void esig::python::init_lie(py::module_ &m) {

    py::options options;
    options.disable_function_signatures();

    pybind11::class_<lie> klass(m, "Lie", LIE_DOC);
    klass.def(py::init(&construct_lie), "data"_a);

    klass.def_property_readonly("width", &lie::width);
    klass.def_property_readonly("max_degree", &lie::depth);
    klass.def_property_readonly("dtype", &lie::coeff_type);
    klass.def_property_readonly("storage_type", &lie::storage_type);

    klass.def("size", &lie::size);
    klass.def("degree", &lie::degree);

    klass.def("__getitem__", [](const lie& self, key_type key) {
        return self[key];
    });
    klass.def("__iter__", [](const lie& self) {
             return py::make_iterator(self.begin(), self.end());
         });

    klass.def("__neg__", &lie::uminus, py::is_operator());

    klass.def("__add__", &lie::add, py::is_operator());
    klass.def("__sub__", &lie::sub, py::is_operator());
    klass.def("__mul__", &lie::smul, py::is_operator());
    klass.def("__truediv__", &lie::smul, py::is_operator());
    klass.def("__mul__", &lie::mul, py::is_operator());
    klass.def("__rmul__", [](const lie& self, const scalars::Scalar & other) { return self.smul(other); },
            py::is_operator());

    klass.def("__mul__", [](const lie& self, scalar_t arg) {
        return self.smul(scalars::Scalar(arg));
    }, py::is_operator());
    klass.def("__mul__", [](const lie& self, long long arg) {
        return self.smul(scalars::Scalar(arg, 1LL, self.coeff_type()));
    }, py::is_operator());
    klass.def("__rmul__", [](const lie& self, scalar_t arg) {
         return self.smul(scalars::Scalar(arg));
    }, py::is_operator());
    klass.def("__rmul__", [](const lie& self, long long arg) {
      return self.smul(scalars::Scalar(arg, 1LL, self.coeff_type()));
    }, py::is_operator());
    klass.def("__truediv__", [](const lie& self, scalar_t arg) {
             return self.sdiv(scalars::Scalar(arg));
         }, py::is_operator());
    klass.def("__truediv__", [](const lie& self, scalar_t arg) {
             return self.sdiv(scalars::Scalar(arg, self.coeff_type()));
         }, py::is_operator());

    klass.def("__iadd__", &lie::add_inplace, py::is_operator());
    klass.def("__isub__", &lie::sub_inplace, py::is_operator());
    klass.def("__imul__", &lie::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &lie::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &lie::mul_inplace, py::is_operator());

    klass.def("__imul__", [](lie& self, scalar_t arg) {
             return self.smul_inplace(scalars::Scalar(arg));
         }, py::is_operator());
    klass.def("__imul__", [](lie& self, long long arg) {
             return self.smul_inplace(scalars::Scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());
    klass.def("__itruediv__", [](lie& self, scalar_t arg) {
             return self.sdiv_inplace(scalars::Scalar(arg));
         }, py::is_operator());
    klass.def("__itruediv__", [](lie& self, long long arg) {
             return self.sdiv_inplace(scalars::Scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());

    klass.def("add_scal_mul", &lie::add_scal_mul, "other"_a, "scalar"_a);
    klass.def("sub_scal_mul", &lie::sub_scal_mul, "other"_a, "scalar"_a);
    klass.def("add_scal_div", &lie::add_scal_div, "other"_a, "scalar"_a);
    klass.def("sub_scal_div", &lie::sub_scal_div, "other"_a, "scalar"_a);

    klass.def("add_mul", &lie::add_mul, "lhs"_a, "rhs"_a);
    klass.def("sub_mul", &lie::sub_mul, "lhs"_a, "rhs"_a);
    klass.def("mul_smul", &lie::mul_smul, "other"_a, "scalar"_a);
    klass.def("mul_sdiv", &lie::mul_sdiv, "other"_a, "scalar"_a);

    klass.def("__str__", [](const lie& self) {
        std::stringstream ss;
        self.print(ss);
        return ss.str();
    });

    klass.def("__repr__", [](const lie& self) {
                std::stringstream ss;
                ss << "Lie(width=" << self.width() << ", depth=" << self.depth();
                ss << ", ctype=" << self.coeff_type()->info().name << ')';
                return ss.str();
            });

    klass.def("__eq__", [](const lie& lhs, const lie& rhs) { return lhs == rhs; });
    klass.def("__neq__", [](const lie& lhs, const lie& rhs) { return lhs != rhs; });

#ifndef ESIG_NO_NUMPY
    klass.def("__array__", [](const lie& arg) {
        py::dtype dtype = esig::python::ctype_to_npy_dtype(arg.coeff_type());

        if (arg.storage_type() == vector_type::dense) {
            auto dense_data = arg.dense_data();
            return py::array(dtype, {dense_data.size()}, {}, dense_data.ptr());
        }
        return py::array(dtype);
    });
#endif
}
