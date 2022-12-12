//
// Created by user on 26/07/22.
//


#include <esig/algebra/python_interface.h>
#include "convert_buffer.h"
#include <esig/pycommon.h>

#include <utility>
#include <algorithm>

namespace py = pybind11;

using esig::dimn_t;
using esig::key_type;


esig::algebra::py_vector_construction_helper
esig::algebra::get_construction_data(const pybind11::object &arg,
                                     const pybind11::kwargs &kwargs) {

    auto helper = esig::algebra::kwargs_to_construction_data(kwargs);

    if (py::isinstance<py::buffer>(arg)) {
        auto info = arg.cast<py::buffer>().request();

        if (info.ndim != 1) {
            throw py::value_error("data has invalid shape");
        }

        auto fmt = esig::py_format_to_esig_format(info.format);


        if (helper.input_vec_type == input_data_type::key_value_array) {
            to_kv_pairs(helper, info);

            helper.begin_ptr = helper.buffer.begin();
            helper.end_ptr = helper.buffer.end();
        } else {
            if (helper.ctype_requested && input_ctype != helper.ctype) {
                convert_buffer(helper.buffer, info, helper.ctype);
                helper.begin_ptr = helper.buffer.begin();
                helper.end_ptr = helper.buffer.end();
            } else {
                helper.ctype = input_ctype;
                helper.begin_ptr = reinterpret_cast<const char *>(info.ptr);
                helper.end_ptr = helper.begin_ptr + info.size * info.itemsize;
            }
            helper.itemsize = size_of(helper.ctype);
            helper.count = info.size;
        }
    } else if (py::isinstance<py::tuple>(arg)) {
        throw py::type_error("conversion from tuple not currently supported");
    }  else if (py::isinstance<py::list>(arg)) {
        convert_list(helper.buffer, arg.cast<py::list>(), helper.ctype);
        helper.count = helper.buffer.size();
        helper.itemsize = helper.buffer.item_size();
        helper.begin_ptr = helper.buffer.begin();
        helper.end_ptr = helper.buffer.end();
    }

    return helper;
}
