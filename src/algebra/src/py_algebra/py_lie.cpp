//
// Created by sam on 03/05/22.
//

#include "py_lie.h"
#include <sstream>
#include "py_iterator.h"
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

using esig::algebra::coefficient_type;


static const char* LIE_DOC = R"eadoc(Element of the free Lie algebra.
)eadoc";


namespace {

esig::algebra::lie lie_from_buffer(const py::object& buf, const py::kwargs& kwargs)
{




}

}


void init_py_lie_iterator(py::module_& m)
{
    using esig::algebra::py_lie_iterator;

    py::class_<py_lie_iterator> klass(m, "PyLieIterator");
    klass.def("__next__", &py_lie_iterator::next);
}

void esig::algebra::init_py_lie(pybind11::module_ &m)
{
    init_py_lie_iterator(m);
    pybind11::class_<lie> klass(m, "Lie", LIE_DOC);

    klass.def_property_readonly("width", &lie::width);
    klass.def_property_readonly("depth", &lie::depth);
    klass.def_property_readonly("dtype", &lie::coeff_type);
    klass.def_property_readonly("storage_type", &lie::storage_type);

    klass.def("size", &lie::size);
    klass.def("degree", &lie::degree);

    klass.def("__getitem__", [](const lie& self, key_type key) {
        return static_cast<scalar_t>(self[key]);
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
    klass.def("__rmul__", [](const lie& self, const coefficient& other) { return self.smul(other); },
            py::is_operator());

    klass.def("__iadd__", &lie::add_inplace, py::is_operator());
    klass.def("__isub__", &lie::sub_inplace, py::is_operator());
    klass.def("__imul__", &lie::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &lie::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &lie::mul_inplace, py::is_operator());

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
                ss << ", ctype=" << static_cast<int>(self.coeff_type()) << ')';
                return ss.str();
            });

    klass.def("__eq__", [](const lie& lhs, const lie& rhs) { return lhs == rhs; });
    klass.def("__neq__", [](const lie& lhs, const lie& rhs) { return lhs != rhs; });

    klass.def("__array__", [](const lie& arg) {
        py::dtype dtype = esig::algebra::dtype_from(arg.coeff_type());

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


}
