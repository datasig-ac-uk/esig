//
// Created by user on 05/05/22.
//

#include "py_common.h"
#include <esig/intervals.h>

#include <pybind11/operators.h>
#include <sstream>

namespace py = pybind11;
using namespace pybind11::literals;

static const char* DYADIC_DOC = R"edoc(A dyadic rational number.
)edoc";


void esig::init_dyadic(py::module_& m)
{
    using esig::dyadic;
    using multiplier_t = typename dyadic::multiplier_t;
    using power_t = typename dyadic::power_t;

    py::class_<dyadic> klass(m, "Dyadic", DYADIC_DOC);

    klass.def(py::init<>());
    klass.def(py::init<multiplier_t>(), "k"_a);
    klass.def(py::init<multiplier_t, power_t>(), "k"_a, "n"_a);

    klass.def("__float__", [](const dyadic& dia) { return static_cast<param_t>(dia); });

    klass.def("rebase", &dyadic::rebase, "resolution"_a);
    klass.def("__str__", [](const dyadic& dia) {
        std::stringstream ss;
        ss << dia;
        return ss.str();
    });
    klass.def("__repr__", [](const dyadic& dia) {
        std::stringstream ss;
        ss << "Dyadic" << dia;
        return ss.str();
    });


    klass.def_static("dyadic_equals", &esig::dyadic_equals, "lhs"_a, "rhs"_a);
    klass.def_static("rational_equals", &esig::rational_equals, "lhs"_a, "rhs"_a);

    klass.def_property_readonly("k", &dyadic::multiplier);
    klass.def_property_readonly("n", &dyadic::power);

    klass.def(py::self < py::self);
    klass.def(py::self <= py::self);
    klass.def(py::self > py::self);
    klass.def(py::self >= py::self);

    klass.def("__iadd__", [](dyadic& dia, multiplier_t val) { return dia.move_forward(val); });

}
