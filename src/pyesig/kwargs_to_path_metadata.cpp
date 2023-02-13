//
// Created by user on 11/02/23.
//

#include "kwargs_to_path_metadata.h"


#include <esig/algebra/context.h>

esig::paths::path_metadata esig::python::kwargs_to_metadata(const py::kwargs &kwargs) {
    paths::path_metadata md{
        0,  // Width
        0,  // Depth
        real_interval {0.0, 1.0}, // Effective domain
        nullptr, // context
        nullptr, // scalar type
        algebra::vector_type::dense // vector type
    };

    if (kwargs.contains("ctx")) {
        md.ctx = kwargs["ctx"].cast<std::shared_ptr<const algebra::context>>();
    }
    if (kwargs.contains("width")) {
        md.width = kwargs["width"].cast<esig::deg_t>();
        if (md.ctx && (md.width != md.ctx->width())) {
            throw py::value_error("width mismatched with provided context");
        }
    } else if (md.ctx) {
        md.width = md.ctx->width();
    }

    if (kwargs.contains("depth")) {
        md.depth = kwargs["depth"].cast<esig::deg_t>();
    } else if (md.ctx) {
        md.depth = md.ctx->depth();
    }

    if (kwargs.contains("ctype")) {
        md.ctype = kwargs["ctype"].cast<py::capsule>().get_pointer<const scalars::scalar_type>();
        if (md.ctx && md.ctype != md.ctx->ctype()) {
            throw py::value_error("mismatch in ctype with provided context");
        }
    } else if (kwargs.contains("dtype")) {
        md.ctype = kwargs["dtype"].cast<py::capsule>().get_pointer<const scalars::scalar_type>();
        if (md.ctx && md.ctype != md.ctx->ctype()) {
            throw py::value_error("mismatch in ctype with provided context");
        }
    } else if (md.ctx) {
        md.ctype = md.ctx->ctype();
    }

    if (kwargs.contains("vtype")) {
        md.result_vec_type = kwargs["vtype"].cast<algebra::vector_type>();
    }

    return md;
}
