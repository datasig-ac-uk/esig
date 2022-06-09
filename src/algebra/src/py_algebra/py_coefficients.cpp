//
// Created by user on 26/05/22.
//

#include "py_coefficients.h"
#include <pybind11/operators.h>
#include <esig/algebra/context.h>
#include <sstream>

static const char* COEFF_DOC = R"eadoc(Type-agnostic scalar coefficient type
)eadoc";

namespace py = pybind11;
using namespace pybind11::literals;
using esig::algebra::coefficient_type;

using esig::algebra::coefficient;

namespace esig {
namespace algebra {
namespace dtl {



}
}
}


template<typename T>
std::shared_ptr<esig::algebra::coefficient_interface>
        coeff_for_ctype(esig::algebra::coefficient_type ctype, T &&arg)
{
#define ESIG_SWITCH_FN(CTYPE) std::shared_ptr<esig::algebra::coefficient_interface>(new \
    esig::algebra::dtl::coefficient_implementation<esig::algebra::type_of_coeff<CTYPE>>(std::forward<T>(arg)))
ESIG_MAKE_CTYPE_SWITCH(ctype)
#undef ESIG_SWITCH_FN
}

void esig::algebra::init_py_coefficients(pybind11::module_ &m)
{
    py::class_<coefficient> klass(m, "Scalar", COEFF_DOC);

    klass.def(py::init<>());
    klass.def(py::init<coefficient_type>(), py::kw_only(), "ctype"_a);
    klass.def(py::init<param_t>(), "arg"_a);
    klass.def(py::init<param_t, coefficient_type>(), "arg"_a, py::kw_only(), "ctype"_a);
    klass.def(py::init<long, long, coefficient_type>(), "numerator"_a, "denominator"_a=1L, py::kw_only(), "ctype"_a);

    klass.def("is_const", &coefficient::is_const);
    klass.def("is_value", &coefficient::is_value);
    klass.def_property_readonly("ctype", &coefficient::ctype);

    klass.def(-py::self);
    klass.def(py::self + py::self);
    klass.def(py::self - py::self);
    klass.def(py::self * py::self);
    klass.def(py::self / py::self);

    klass.def(py::self += py::self);
    klass.def(py::self -= py::self);
    klass.def(py::self *= py::self);
    klass.def(py::self /= py::self);

    klass.def("__add__", [](const coefficient& lhs, double rhs) { return lhs + coefficient(rhs); }, py::is_operator());
    klass.def("__sub__", [](const coefficient& lhs, double rhs) { return lhs - coefficient(rhs); }, py::is_operator());
    klass.def("__mul__", [](const coefficient& lhs, double rhs) { return lhs * coefficient(rhs); }, py::is_operator());
    klass.def("__div__", [](const coefficient& lhs, double rhs) { return lhs / coefficient(rhs); }, py::is_operator());

    klass.def("__radd__", [](const coefficient& rhs, double lhs) { return coefficient(lhs) + rhs; }, py::is_operator());
    klass.def("__rsub__", [](const coefficient& rhs, double lhs) { return coefficient(lhs) - rhs; }, py::is_operator());
    klass.def("__rmul__", [](const coefficient& rhs, double lhs) { return coefficient(lhs) * rhs; }, py::is_operator());
    klass.def("__rdiv__", [](const coefficient& rhs, double lhs) { return coefficient(lhs) / rhs; }, py::is_operator());

    klass.def("__repr__", [](const coefficient& arg) {
             std::stringstream ss;
             ss << "Scalar(ctype=" << arg.ctype() << ')';
         });
    klass.def("__str__", [](const coefficient& arg) {
             std::stringstream ss;
             //ss << arg;
             return ss.str();
         });

    klass.def("__float__", [](const coefficient& arg) { return double(arg); });

}
