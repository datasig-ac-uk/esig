//
// Created by user on 08/12/22.
//

#include "py_paths.h"

#include <esig/algebra/context.h>
#include <esig/paths/path.h>

using namespace esig;
using namespace esig::python;

using namespace pybind11::literals;

using algebra::context;
using stream = paths::path;

using ivl_lsig_fn = algebra::lie (stream::*)(const interval&, accuracy_t) const;
using ivl_ctx_lsig_fn = algebra::lie (stream::*)(const interval&, accuracy_t, const context&) const;
using lsig_fn = algebra::lie (stream::*)(accuracy_t) const;
using ctx_lsig_fn = algebra::lie (stream::*)(accuracy_t, const context&) const;

using ivl_sig_fn = algebra::free_tensor (stream::*)(const interval&, accuracy_t) const;
using ivl_ctx_sig_fn = algebra::free_tensor (stream::*)(const interval&, accuracy_t, const context&) const;
using sig_fn = algebra::free_tensor (stream::*)(accuracy_t) const;
using ctx_sig_fn = algebra::free_tensor (stream::*)(accuracy_t, const context&) const;

using ivl_sigder_fn = algebra::free_tensor (stream::*)(const interval& ivl, const algebra::lie&, accuracy_t) const;
using sigder_fn = algebra::free_tensor (stream::*)(const typename paths::path::perturbation_list_t&, accuracy_t) const;
using ctx_sigder_fn = algebra::free_tensor (stream::*)(const typename paths::path::perturbation_list_t&, accuracy_t, const context&) const;


void esig::python::init_paths(py::module_ &m) {


    py::class_<stream> klass(m, "Stream");

    klass.def("signature", static_cast<sig_fn>(&stream::signature), "accuracy"_a);
    klass.def("signature", static_cast<ctx_sig_fn>(&stream::signature), "accuracy"_a, "context"_a);
    klass.def("signature", static_cast<ivl_sig_fn>(&stream::signature), "domain"_a, "accuracy"_a);
    klass.def("signature", static_cast<ivl_ctx_sig_fn>(&stream::signature), "domain"_a, "accuracy"_a, "context"_a);

    klass.def("log_signature", static_cast<lsig_fn>(&stream::log_signature), "accuracy"_a);
    klass.def("log_signature", static_cast<ctx_lsig_fn>(&stream::log_signature), "accuracy"_a, "context"_a);
    klass.def("log_signature", static_cast<ivl_lsig_fn>(&stream::log_signature), "domain"_a, "accuracy"_a);
    klass.def("log_signature", static_cast<ivl_ctx_lsig_fn>(&stream::log_signature), "domain"_a, "accuracy"_a, "context"_a);

    klass.def("signature_derivative", static_cast<ivl_sigder_fn>(&stream::signature_derivative), "domain"_a, "perturbation"_a, "accuracy"_a);
    klass.def("signature_derivative", static_cast<sigder_fn>(&stream::signature_derivative), "perturbations"_a, "accuracy"_a);
    klass.def("signature_derivative", static_cast<ctx_sigder_fn>(&stream::signature_derivative), "perturbations"_a, "accuracy"_a, "context"_a);


}
