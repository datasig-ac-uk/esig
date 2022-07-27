//
// Created by user on 25/07/22.
//
#include <esig/algebra/python_interface.h>
#include "convert_buffer.h"

namespace py = pybind11;

using esig::key_type;
using esig::dimn_t;




void copy_key_from_pybuffer(std::vector<char>& buffer, const py::buffer_info& info)
{
    buffer.resize(info.size*sizeof(key_type));
    using esig::algebra::copy_convert;

    switch (info.format[0]) {
        case 'b':
        case 'B':
            copy_convert<key_type, unsigned char>(buffer.data(), info.ptr, info.size);
            break;
        case 'h':
        case 'H':
            copy_convert<key_type, unsigned short>(buffer.data(), info.ptr, info.size);
            break;
        case 'i':
        case 'I':
            copy_convert<key_type, unsigned int>(buffer.data(), info.ptr, info.size);
            break;
        case 'l':
        case 'k':
            copy_convert<key_type, unsigned long>(buffer.data(), info.ptr, info.size);
            break;
        case 'L':
        case 'K':
            copy_convert<key_type, unsigned long long>(buffer.data(), info.ptr, info.size);
            break;
        case 'n':
            copy_convert<key_type, py::ssize_t>(buffer.data(), info.ptr, info.size);
            break;
        default:
            throw py::type_error("invalid key conversion type");
    }

}


esig::algebra::py_vector_construction_helper
esig::algebra::kwargs_to_construction_data(const pybind11::kwargs &kwargs)
{
   py_vector_construction_helper helper;

    if (kwargs.contains("context")) {
        helper.ctx = kwargs["context"].cast<std::shared_ptr<esig::algebra::context>>();
        helper.width = helper.ctx->width();
        helper.depth = helper.ctx->depth();
        helper.ctype = helper.ctx->ctype();
        helper.ctype_requested = true;
    }

    if (!helper.ctx && kwargs.contains("ctype")) {
        helper.ctype = kwargs["ctype"].cast<coefficient_type>();
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
        helper.vtype = kwargs["vector_type"].cast<vector_type>();
        helper.vtype_requested = true;
    }

    if (kwargs.contains("keys")) {
        const auto& arg = kwargs["keys"];
        if (py::isinstance<key_type>(arg)) {
            helper.buffer.resize(sizeof(key_type));
            *reinterpret_cast<key_type*>(helper.buffer.data()) = arg.cast<key_type>();
        } else if (py::isinstance<py::buffer>(arg)) {
            auto key_info = arg.cast<py::buffer>().request();
            helper.buffer.resize(key_info.size*sizeof(key_type));
            copy_key_from_pybuffer(helper.buffer, key_info);
        }
    }



}
