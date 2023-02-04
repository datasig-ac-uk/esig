//
// Created by user on 09/12/22.
//

#include "lie.h"

#include <pybind11/operators.h>

#ifndef ESIG_NO_NUMPY
#include <pybind11/numpy.h>
#endif

#include <esig/scalars.h>
#include <esig/algebra/lie_interface.h>

#include "ctype_to_npy_dtype.h"
#include "py_lie_key.h"

using namespace esig;
using namespace esig::algebra;
using namespace esig::python;
using namespace pybind11::literals;


static const char* LIE_DOC = R"edoc(
Element of the free Lie algebra.
)edoc";



class py_lie_iterator
{
    algebra_iterator m_it;
    algebra_iterator m_end;

public:
    py_lie_iterator(algebra_iterator it, algebra_iterator end);
    std::pair<py_lie_key, scalars::scalar> next();

};

py_lie_iterator::py_lie_iterator(algebra_iterator it, algebra_iterator end) {
}
std::pair<py_lie_key, scalars::scalar> py_lie_iterator::next() {
    if (m_it == m_end) {
        throw py::stop_iteration();
    }
    return {{m_it.get_context(), m_it->key()}, m_it->value()};
}

static void init_py_lie_iterator(py::module_& m)
{

    py::class_<py_lie_iterator> klass(m, "PyLieIterator");
//    klass.def("__next__", &py_lie_iterator::next);
}


void esig::python::init_lie(py::module_ &m) {

    py::options options;
    options.disable_function_signatures();

    init_py_lie_iterator(m);
    pybind11::class_<lie> klass(m, "Lie", LIE_DOC);

    klass.def_property_readonly("width", &lie::width);
    klass.def_property_readonly("depth", &lie::depth);
    klass.def_property_readonly("dtype", &lie::coeff_type);
    klass.def_property_readonly("storage_type", &lie::storage_type);

    klass.def("size", &lie::size);
    klass.def("degree", &lie::degree);

    klass.def("__getitem__", [](const lie& self, key_type key) {
        return self[key];
    });
    klass.def("__iter__", [](const lie& self) {
             return py_lie_iterator(self.begin(), self.end());
         });

    klass.def("__neg__", &lie::uminus, py::is_operator());

    klass.def("__add__", &lie::add, py::is_operator());
    klass.def("__sub__", &lie::sub, py::is_operator());
    klass.def("__mul__", &lie::smul, py::is_operator());
    klass.def("__truediv__", &lie::smul, py::is_operator());
    klass.def("__mul__", &lie::mul, py::is_operator());
    klass.def("__rmul__", [](const lie& self, const scalars::scalar& other) { return self.smul(other); },
            py::is_operator());

    klass.def("__mul__", [](const lie& self, scalar_t arg) {
        return self.smul(scalars::scalar(arg));
    }, py::is_operator());
    klass.def("__mul__", [](const lie& self, long long arg) {
        return self.smul(scalars::scalar(arg, 1LL, self.coeff_type()));
    }, py::is_operator());
    klass.def("__rmul__", [](const lie& self, scalar_t arg) {
         return self.smul(scalars::scalar(arg));
    }, py::is_operator());
    klass.def("__rmul__", [](const lie& self, long long arg) {
      return self.smul(scalars::scalar(arg, 1LL, self.coeff_type()));
    }, py::is_operator());
    klass.def("__truediv__", [](const lie& self, scalar_t arg) {
             return self.sdiv(scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__truediv__", [](const lie& self, scalar_t arg) {
             return self.sdiv(scalars::scalar(arg, self.coeff_type()));
         }, py::is_operator());

    klass.def("__iadd__", &lie::add_inplace, py::is_operator());
    klass.def("__isub__", &lie::sub_inplace, py::is_operator());
    klass.def("__imul__", &lie::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &lie::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &lie::mul_inplace, py::is_operator());

    klass.def("__imul__", [](lie& self, scalar_t arg) {
             return self.smul_inplace(scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__imul__", [](lie& self, long long arg) {
             return self.smul_inplace(scalars::scalar(arg, 1LL, self.coeff_type()));
         }, py::is_operator());
    klass.def("__itruediv__", [](lie& self, scalar_t arg) {
             return self.sdiv_inplace(scalars::scalar(arg));
         }, py::is_operator());
    klass.def("__itruediv__", [](lie& self, long long arg) {
             return self.sdiv_inplace(scalars::scalar(arg, 1LL, self.coeff_type()));
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
#if 0
    klass.def("__array__", [](const lie& arg) {
        py::dtype dtype = esig::python::ctype_to_npy_dtype(arg.coeff_type());

        if (arg.storage_type() == vector_type::dense) {
            auto it = arg.iterate_dense_components().next();
            if (!it) {
                throw std::runtime_error("dense data should be valid");
            }
            const auto *ptr = it.begin();
            return py::array(dtype, {arg.size()}, {}, ptr);
        }
        return py::array(dtype);
    });
#endif
#endif
}
