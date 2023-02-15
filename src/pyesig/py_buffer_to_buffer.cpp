//
// Created by user on 09/12/22.
//

#include "py_buffer_to_buffer.h"

#include "py_fmt_to_esig_fmt.h"

using namespace esig;
using namespace esig::scalars;


esig::scalars::owned_scalar_array esig::python::py_buffer_to_buffer(const py::buffer_info &buf, const esig::scalars::scalar_type *& type) {
    auto format = esig::python::py_fmt_to_esig_fmt(buf.format);

    if (type == nullptr) {
        if (format == "f64") {
            type = dtl::scalar_type_holder<double>::get_type();
        } else if (format == "f32") {
            type = dtl::scalar_type_holder<float>::get_type();
        } else {
            type = dtl::scalar_type_holder<double>::get_type();
        }
    }

    owned_scalar_array out_array(type, buf.size);
    type->convert_copy(out_array, buf.ptr, static_cast<dimn_t>(buf.size), format);
    return out_array;
}
