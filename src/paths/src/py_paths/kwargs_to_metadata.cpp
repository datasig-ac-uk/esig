//
// Created by user on 22/07/22.
//

#include <esig/paths/python_interface.h>


esig::paths::path_metadata
esig::paths::process_kwargs_to_metadata(const pybind11::kwargs& kwargs)
{
    deg_t width;
    deg_t depth;
    real_interval effective_domain;

    algebra::coefficient_type ctype;
    algebra::vector_type input_vec_type;
    algebra::vector_type result_vec_type;

    return path_metadata {
        width, depth, effective_domain, ctype, input_vec_type, result_vec_type
    };
}
