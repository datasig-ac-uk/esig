//
// Created by user on 05/05/22.
//


#include "py_common.h"
#include <esig/intervals.h>


namespace py = pybind11;
using namespace pybind11::literals;

static const char* DYADIC_INTERVAL_DOC = R"edoc(A dyadic interval.)edoc";

static const char* TO_DYADIC_INT_DOC = R"edoc(Dissect an interval into a partition of dyadic intervals.
)edoc";


void esig::init_dyadic_interval(pybind11::module_ &m)
{
    using esig::dyadic;
    using esig::dyadic_interval;
    using multiplier_t = typename dyadic::multiplier_t;
    using power_t = typename dyadic::power_t;

    py::options options;
    options.disable_function_signatures();

    py::class_<dyadic_interval, esig::interval, dyadic>
            klass(m, "DyadicInterval", DYADIC_INTERVAL_DOC);

    klass.def(py::init<>());
    klass.def(py::init<esig::interval_type>(), "interval_type"_a);
    klass.def(py::init<multiplier_t>(), "k"_a);
    klass.def(py::init<multiplier_t, power_t>(), "k"_a, "n"_a);
    klass.def(py::init<multiplier_t, power_t, esig::interval_type>(),
            "k"_a, "n"_a, "interval_type"_a);
    klass.def(py::init<dyadic>(), "dyadic"_a);
    klass.def(py::init<dyadic, power_t>(), "dyadic"_a, "resolution"_a);

    klass.def("dyadic_included_end", &dyadic_interval::dincluded_end);
    klass.def("dyadic_excluded_end", &dyadic_interval::dexcluded_end);
    klass.def("dyadic_inf", &dyadic_interval::dinf);
    klass.def("dyadic_sup", &dyadic_interval::dsup);

    klass.def("shrink_to_contained_end", &dyadic_interval::shrink_to_contained_end, "arg"_a=1);
    klass.def("shrink_to_omitted_end", &dyadic_interval::shrink_to_omitted_end);
    klass.def("shrink_left", &dyadic_interval::shrink_interval_left);
    klass.def("shrink_right", &dyadic_interval::shrink_interval_right);


    klass.def_static("to_dyadic_intervals", &esig::to_dyadic_intervals, "inf"_a, "sup"_a,
                     "resolution"_a, "interval_type"_a, TO_DYADIC_INT_DOC);
    klass.def_static("to_dyadic_intervals",
            [](const esig::interval& domain, power_t resolution, esig::interval_type itype)
            { return esig::to_dyadic_intervals(domain.inf(), domain.sup(), resolution, itype); },
            "domain"_a, "resolution"_a, "interval_type"_a
            );

}
