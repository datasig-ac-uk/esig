//
// Created by sam on 19/08/22.
//

#include "kwargs_to_context.h"

namespace py = pybind11;
using namespace esig;

std::shared_ptr<const algebra::context> paths::kwargs_to_context(const esig::paths::path_metadata& md, const pybind11::kwargs &kwargs)
{
    if (kwargs.empty()) {
        return { nullptr };
    }

    if (kwargs.contains("context")) {
        auto ctx = kwargs["context"].cast<std::shared_ptr<algebra::context>>();
        if (ctx->width() != md.width) {
            throw py::value_error("mismatched width between path and provided context");
        }
        return ctx;
    }

    deg_t depth = md.depth;
    algebra::coefficient_type ctype = md.ctype;

    if (kwargs.contains("depth")) {
        depth = kwargs["depth"].cast<deg_t>();
        if (depth == md.depth) {
            return { nullptr };
        }
    }

    if (kwargs.contains("ctype")) {
        ctype = kwargs["ctype"].cast<algebra::coefficient_type>();
    }

    return algebra::get_context(md.width, depth, ctype);
}
