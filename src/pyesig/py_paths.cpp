//
// Created by user on 08/12/22.
//

#include "py_paths.h"

#include <pybind11/stl.h>

#include <esig/algebra/context.h>
#include <esig/paths/lie_increment_path.h>
#include <esig/paths/path.h>
#include <esig/paths/piecewise_lie_path.h>

#include "kwargs_to_path_metadata.h"
#include "py_scalars.h"

using namespace esig;
using namespace esig::python;

using namespace pybind11::literals;

using algebra::context;
using stream = paths::path;

using ivl_lsig_fn = algebra::lie (stream::*)(const interval &, accuracy_t) const;
using ivl_ctx_lsig_fn = algebra::lie (stream::*)(const interval &, accuracy_t, const context &) const;
using lsig_fn = algebra::lie (stream::*)(accuracy_t) const;
using ctx_lsig_fn = algebra::lie (stream::*)(accuracy_t, const context &) const;

using ivl_sig_fn = algebra::free_tensor (stream::*)(const interval &, accuracy_t) const;
using ivl_ctx_sig_fn = algebra::free_tensor (stream::*)(const interval &, accuracy_t, const context &) const;
using sig_fn = algebra::free_tensor (stream::*)(accuracy_t) const;
using ctx_sig_fn = algebra::free_tensor (stream::*)(accuracy_t, const context &) const;

using ivl_sigder_fn = algebra::free_tensor (stream::*)(const interval &ivl, const algebra::lie &, accuracy_t) const;
using sigder_fn = algebra::free_tensor (stream::*)(const typename paths::path::perturbation_list_t &, accuracy_t) const;
using ctx_sigder_fn = algebra::free_tensor (stream::*)(const typename paths::path::perturbation_list_t &, accuracy_t, const context &) const;

void esig::python::buffer_to_indices(std::vector<param_t> &indices, const py::buffer_info &info) {
    auto count = info.size;
    const auto *ptr = info.ptr;

    indices.resize(count);
    auto *dst = indices.data();
    if (info.format[0] == 'd') {
        memcpy(dst, ptr, count * sizeof(double));
    } else {
        auto conversion = scalars::get_conversion(py_buffer_to_type_id(info), "f64");
        conversion(scalars::ScalarPointer{dst, nullptr}, scalars::ScalarPointer{ptr, nullptr}, count);
    }
}

stream lie_increment_path_from_increments(const py::object &data, const py::kwargs &kwargs) {
    auto md = kwargs_to_metadata(kwargs);

    std::vector<param_t> indices;

    python::py_to_buffer_options options;
    options.type = md.ctype;
    options.max_nested = 2;
    options.allow_scalar = false;

    auto buffer = python::py_to_buffer(data, options);

    idimn_t increment_size = 0;
    idimn_t num_increments = 0;

    if (options.shape.empty()) {
        increment_size = buffer.size();
        num_increments = 1;
    } else if (options.shape.size() == 1) {
        increment_size = options.shape[0];
        num_increments = 1;
    } else {
        increment_size = options.shape[1];
        num_increments = options.shape[0];
    }

    if (md.ctype == nullptr) {
        if (options.type != nullptr) {
            md.ctype = options.type;
        } else {
            throw py::type_error("unable to deduce suitable scalar type");
        }
    }

    assert(buffer.size() == static_cast<dimn_t>(increment_size * num_increments));
    assert(md.ctype != nullptr);
    if (!md.ctx) {
        if (md.width == 0 || md.depth == 0) {
            throw py::value_error("either ctx or both width and depth must be specified");
        }
        md.ctx = algebra::get_context(md.width, md.depth, md.ctype);
    }

    if (kwargs.contains("indices")) {
        auto indices_arg = kwargs["indices"];

        if (py::isinstance<py::buffer>(indices_arg)) {
            auto info = py::reinterpret_borrow<py::buffer>(indices_arg).request();
            esig::python::buffer_to_indices(indices, info);
        } else if (py::isinstance<py::int_>(indices_arg)) {
            // Interpret this as a column in the data;
            auto icol = indices_arg.cast<idimn_t>();
            if (icol < 0) {
                icol += increment_size;
            }
            if (icol < 0 || icol >= increment_size) {
                throw py::value_error("index out of bounds");
            }

            indices.reserve(num_increments);
            for (idimn_t i = 0; i < num_increments; ++i) {
                indices.push_back(static_cast<param_t>(buffer[i * increment_size + icol].to_scalar_t()));
            }
        } else if (py::isinstance<py::sequence>(indices_arg)) {
            indices = indices_arg.cast<std::vector<param_t>>();
        }
    }

    if (indices.empty()) {
        indices.reserve(num_increments);
        for (dimn_t i = 0; i < num_increments; ++i) {
            indices.emplace_back(i);
        }
    } else if (indices.size() != num_increments) {
        throw py::value_error("mismatch between number of rows in data and number of indices");
    }

    auto result = stream(paths::lie_increment_path(scalars::OwnedScalarArray(buffer), indices, md));

    if (options.cleanup) {
        options.cleanup();
    }

    return result;
}

static stream lie_increment_path_from_values(py::object data, py::kwargs kwargs) {

    return stream();
}

void esig::python::init_paths(py::module_ &m) {

    py::class_<stream> klass(m, "Stream");

    klass.def_property_readonly("width", [](const stream &stream) { return stream.metadata().width; });
    klass.def_property_readonly("depth", [](const stream &stream) { return stream.metadata().depth; });

    klass.def("signature", static_cast<sig_fn>(&stream::signature), "accuracy"_a);
    klass.def("signature", static_cast<ctx_sig_fn>(&stream::signature), "accuracy"_a, "context"_a);
    klass.def("signature", static_cast<ivl_sig_fn>(&stream::signature), "domain"_a, "accuracy"_a);
    klass.def("signature", static_cast<ivl_ctx_sig_fn>(&stream::signature), "domain"_a, "accuracy"_a, "context"_a);
    klass.def(
        "signature", [](const stream &self, param_t inf, param_t sup, accuracy_t accuracy) {
            return self.signature(real_interval(inf, sup), accuracy);
        },
        "inf"_a, "sup"_a, "accuracy"_a);
    klass.def(
        "signature", [](const stream &self, param_t inf, param_t sup, accuracy_t accuracy, const context &ctx) {
            return self.signature(real_interval(inf, sup), accuracy, ctx);
        },
        "inf"_a, "sup"_a, "accuracy"_a, "ctx"_a);
    klass.def(
        "signature", [](const stream &self, param_t inf, param_t sup, accuracy_t accuracy, deg_t depth) {
            auto ctx = self.metadata().ctx->get_alike(depth);
            return self.signature(real_interval(inf, sup), accuracy, *ctx);
        },
        "inf"_a, "sup"_a, "accuracy"_a, py::kw_only(), "depth"_a);

    klass.def("log_signature", static_cast<lsig_fn>(&stream::log_signature), "accuracy"_a);
    klass.def("log_signature", static_cast<ctx_lsig_fn>(&stream::log_signature), "accuracy"_a, "context"_a);
    klass.def("log_signature", static_cast<ivl_lsig_fn>(&stream::log_signature), "domain"_a, "accuracy"_a);
    klass.def("log_signature", static_cast<ivl_ctx_lsig_fn>(&stream::log_signature), "domain"_a, "accuracy"_a, "context"_a);

    klass.def("signature_derivative", static_cast<ivl_sigder_fn>(&stream::signature_derivative), "domain"_a, "perturbation"_a, "accuracy"_a);
    klass.def("signature_derivative", static_cast<sigder_fn>(&stream::signature_derivative), "perturbations"_a, "accuracy"_a);
    klass.def("signature_derivative", static_cast<ctx_sigder_fn>(&stream::signature_derivative), "perturbations"_a, "accuracy"_a, "context"_a);
    klass.def(
        "signature_derivative", [](const stream &self, const interval &domain, const algebra::lie &perturbation, accuracy_t accuracy, deg_t depth) {
            auto ctx = self.metadata().ctx->get_alike(depth);
            return self.signature_derivative(domain, perturbation, accuracy, *ctx);
        },
        "domain"_a, "perturbation"_a, "accuracy"_a, py::kw_only(), "depth"_a);

    py::class_<esig::paths::lie_increment_path> LieIncrementPath(m, "LieIncrementPath");

    LieIncrementPath.def_static("from_increments", &lie_increment_path_from_increments, "data"_a);
    LieIncrementPath.def_static("from_values", &lie_increment_path_from_values, "data"_a);

    py::class_<esig::paths::piecewise_lie_path> PiecewiseLiePath(m, "PiecewiseLiePath");
}
