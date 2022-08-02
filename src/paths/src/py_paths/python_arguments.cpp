//
// Created by user on 28/07/22.
//

#include "python_arguments.h"

namespace py = pybind11;

esig::paths::path_metadata esig::paths::parse_kwargs_to_metadata(const pybind11::kwargs &kwargs, esig::paths::additional_args args) {
    path_metadata md;

    if (kwargs.contains("context")) {
        md.ctx = kwargs["context"].cast<std::shared_ptr<algebra::context>>();
    } else if (kwargs.contains("ctx")) {
        md.ctx = kwargs["ctx"].cast<std::shared_ptr<algebra::context>>();
    }

    // Provided context takes precedence over other arguments.
    if (md.ctx) {
        md.width = md.ctx->width();
        md.depth = md.ctx->depth();
        md.ctype = md.ctx->ctype();
    } else {
        // If the context is not set we need to get the width/depth/ctype
        // from the arguments or from the additional info provided.

        if (kwargs.contains("width") && !kwargs["width"].is_none()) {
            md.width = kwargs["width"].cast<deg_t>();
        } else {
            // requested args takes precedence over derived args
            md.width = args.width;
        }
        if (kwargs.contains("depth") && !kwargs["depth"].is_none()) {
            md.depth = kwargs["depth"].cast<deg_t>();
        } else if (args.depth != 0){
            // requested args takes precedence over derived args
            md.depth = args.depth;
        } else {
            md.depth = 2;
        }
        // Either ctype or dtype are acceptable aliases for the coefficient
        // type
        if (kwargs.contains("ctype") && !kwargs["ctype"].is_none()) {
            md.ctype = kwargs["ctype"].cast<algebra::coefficient_type>();
        } else if (kwargs.contains("dtype")) {
            md.ctype = kwargs["dtype"].cast<algebra::coefficient_type>();
        } else {
            md.ctype = args.ctype;
        }

        if (md.width == 0) {
            throw py::value_error("could not deduce underlying path dimension");
        }

        md.ctx = algebra::get_context(md.width, md.depth, md.ctype);
    }

    // Check that the settings we've been provided are compatible with
    // the data we've received?


    // set defaults for the domain, input data type, result vector_type
    if (kwargs.contains("domain")) {
        md.effective_domain = kwargs.cast<real_interval>();
    } else {
        md.effective_domain = args.domain;
    }
    if (kwargs.contains("vtype")) {
        md.result_vec_type = kwargs["vtype"].cast<algebra::vector_type>();
    } else {
        md.result_vec_type = args.result_vtype;
    }
    md.data_type = args.data_type;


    return md;
}
