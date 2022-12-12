//
// Created by user on 09/12/22.
//

#include "kwargs_to_vector_construction.h"

#include <memory>

#include <esig/algebra/context.h>

#include "py_arg_to_ctype.h"

esig::python::py_vector_construction_helper esig::python::kwargs_to_construction_data(const py::kwargs &kwargs) {

    py_vector_construction_helper helper;

    if (kwargs.contains("context")) {
        helper.ctx = kwargs["context"].cast<std::shared_ptr<algebra::context>>();
        helper.width = helper.ctx->width();
        helper.depth = helper.ctx->depth();
        helper.ctype = helper.ctx->ctype();
        helper.ctype_requested = true;
    }

    if (!helper.ctx && kwargs.contains("ctype")) {
        helper.ctype = py_arg_to_ctype(kwargs["ctype"]);
        helper.ctype_requested = true;
    }

    if (!helper.ctx && kwargs.contains("depth")) {
        helper.depth = kwargs["depth"].cast<deg_t>();
    } else {
        helper.depth = 2;
    }

    if (!helper.ctx && kwargs.contains("width")) {
        helper.width = kwargs["width"].cast<deg_t>();
        helper.ctx = esig::algebra::get_context(helper.width, helper.depth, helper.ctype);
    }

    if (kwargs.contains("vector_type")) {
        helper.vtype = kwargs["vector_type"].cast<algebra::vector_type>();
        helper.vtype_requested = true;
    }

    if (kwargs.contains("keys")) {
        const auto &arg = kwargs["keys"];
        if (py::isinstance<key_type>(arg)) {
        } else if (py::isinstance<py::buffer>(arg)) {
            auto key_info = arg.cast<py::buffer>().request();
        }
    }

    return helper;
}
