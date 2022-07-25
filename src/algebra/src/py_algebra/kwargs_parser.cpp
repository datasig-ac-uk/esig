//
// Created by user on 25/07/22.
//
#include <esig/algebra/python_interface.h>



esig::algebra::py_vector_construction_helper
esig::algebra::kwargs_to_construction_data(const pybind11::kwargs &kwargs)
{
    deg_t width = 0, depth = 0;
    std::shared_ptr<context> ctx(nullptr);
    vector_type vtype = vector_type::dense;
    coefficient_type ctype = coefficient_type::dp_real;
    bool ctype_requested = false;
    bool vtype_requested = false;

    if (kwargs.contains("context")) {
        ctx = kwargs["context"].cast<std::shared_ptr<esig::algebra::context>>();
        width = ctx->width();
        depth = ctx->depth();
        ctype = ctx->ctype();
        ctype_requested = true;
    }

    if (!ctx && kwargs.contains("ctype")) {
        ctype = kwargs["ctype"].cast<coefficient_type>();
        ctype_requested = true;
    }

    if (!ctx && kwargs.contains("depth")) {
        depth = kwargs["depth"].cast<deg_t>();
    } else {
        depth = 2;
    }

    if (!ctx && kwargs.contains("width")) {
        width = kwargs["width"].cast<deg_t>();
        ctx = esig::algebra::get_context(width, depth, ctype);
    }

    if (kwargs.contains("vector_type")) {
        vtype = kwargs["vector_type"].cast<vector_type>();
        vtype_requested = true;
    }


    return {ctx, width, depth, ctype, vtype, ctype_requested, vtype_requested};

}
