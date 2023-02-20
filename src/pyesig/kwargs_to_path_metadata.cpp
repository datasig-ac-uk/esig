//
// Created by user on 11/02/23.
//

#include "kwargs_to_path_metadata.h"

#include "py_context.h"
#include "scalar_meta.h"
#include "py_arg_to_ctype.h"
#include "numpy.h"

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
        md.ctx = kwargs["ctx"].cast<py_context>().get_pointer();
        md.width = md.ctx->width();
        md.depth = md.ctx->depth();
        md.ctype = md.ctx->ctype();
    } else {

        if (kwargs.contains("width")) {
            md.width = kwargs["width"].cast<esig::deg_t>();
        }
        if (kwargs.contains("depth")) {
            md.depth = kwargs["depth"].cast<esig::deg_t>();
        }
        if (kwargs.contains("ctype")) {
            md.ctype = esig::python::py_arg_to_ctype(kwargs["ctype"]);
        }
#ifndef ESIG_NO_NUMPY
        else if (kwargs.contains("dtype")) {
            auto dtype = kwargs["dtype"];
            if (py::isinstance<py::dtype>(dtype)) {
                md.ctype = npy_dtype_to_ctype(dtype);
            } else {
                md.ctype = py_arg_to_ctype(dtype);
            }
        }
#endif
    }

    if (kwargs.contains("vtype")) {
        md.result_vec_type = kwargs["vtype"].cast<algebra::vector_type>();
    }

    return md;
}
