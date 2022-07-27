//
// Created by user on 26/07/22.
//


#include <esig/algebra/python_interface.h>
#include "convert_buffer.h"

#include <utility>
#include <algorithm>

namespace py = pybind11;

using esig::dimn_t;
using esig::key_type;
using esig::algebra::coefficient_type;


template<coefficient_type CType>
void convert_buffer_impl(esig::algebra::allocating_data_buffer& buffer, const py::buffer_info &info) {
    using scalar_type = esig::algebra::type_of_coeff<CType>;
    buffer.set_allocator_and_alloc(esig::algebra::allocator_for_coeff(CType), info.size);

    using esig::algebra::copy_convert;

    switch (info.format[0]) {
        case 'f':
            copy_convert<scalar_type, float>(buffer.begin(), info.ptr, info.size);
            break;
        case 'd':
            copy_convert<scalar_type, double>(buffer.begin(), info.ptr, info.size);
            break;
    }
}

template <coefficient_type CType>
void to_kv_pairs_impl(esig::algebra::py_vector_construction_helper& helper, const py::buffer_info& info) {
    using scalar_type = esig::algebra::type_of_coeff<CType>;
    using kv_t = std::pair<key_type, scalar_type>;
    auto& buffer = helper.buffer;

    const dimn_t n_keys = buffer.size() / sizeof(key_type);
    // Buffer is assumed to contain the keys that will be used, so the first task is to
    // make a copy that contains just the keys as key_types
    esig::algebra::allocating_data_buffer key_buffer;
    buffer.swap(key_buffer);

    // Now increase the size of the vector to accommodate pairs rather than
    // just keys
    if (n_keys != info.size) {
        throw py::value_error("mismatching number of keys and values");
    }
    buffer.set_allocator_and_alloc(esig::algebra::allocator_for_key_coeff(CType), n_keys);

    using esig::algebra::copy_kv_convert;

    // copy over the key, values
    switch (info.format[0]) {
        case 'f':
            copy_kv_convert<kv_t, key_type, float>(buffer.begin(), key_buffer.begin(), info.ptr, info.size);
            break;
        case 'd':
            copy_kv_convert<kv_t, key_type, double>(buffer.begin(), key_buffer.begin(), info.ptr, info.size);
            break;
        default:
            throw py::type_error("unsupported data type");
    }

    std::sort(reinterpret_cast<kv_t*>(buffer.begin()),
              reinterpret_cast<kv_t*>(buffer.end()),
              [](const kv_t& a, const kv_t& b) { return a.first < b.first; });

    // set the remaining stuff
    helper.itemsize = sizeof(kv_t);
    helper.count = n_keys;

    /*
     * Since we are dealing with k-v data, we can probably deduce the width
     * by looking at the maximum key if we haven't already got a width. The
     * pairs are sorted so they are in key order, so we only need to look
     * at the last element.
     */
    if (!helper.ctx && n_keys > 0) {
        const auto *back = reinterpret_cast <const kv_t*>(buffer.end());
        helper.width = dimn_t((--back)->first);
        if (helper.width != 0) {
            // do not handle width == 0 case
            helper.ctx = esig::algebra::get_context(helper.width, helper.depth, helper.ctype);
        }
    }
}

template <coefficient_type CType>
void convert_list_impl(esig::algebra::allocating_data_buffer& buffer, const py::list& info)
{
    using scalar_type = esig::algebra::type_of_coeff<CType>;
    buffer.set_allocator_and_alloc(esig::algebra::allocator_for_coeff(CType), info.size());

    auto* ptr = reinterpret_cast<scalar_type *>(buffer.begin());
    for (const auto& item : info) {
        *(ptr++) = item.cast<scalar_type>();
    }

}


void convert_buffer(esig::algebra::allocating_data_buffer& buffer, const py::buffer_info &info, coefficient_type ctype) {
#define ESIG_SWITCH_FN(CTYPE) convert_buffer_impl<CTYPE>(buffer, info)
    ESIG_MAKE_CTYPE_SWITCH(ctype)
#undef ESIG_SWITCH_FN
}

void convert_list(esig::algebra::allocating_data_buffer& buffer, const py::list& info, coefficient_type ctype) {
#define ESIG_SWITCH_FN(CTYPE) convert_list_impl<CTYPE>(buffer, info)
    ESIG_MAKE_CTYPE_SWITCH(ctype)
#undef ESIG_SWITCH_FN
}


void to_kv_pairs(esig::algebra::py_vector_construction_helper& helper, const py::buffer_info& info)
{
#define ESIG_SWITCH_FN(CTYPE) to_kv_pairs_impl<CTYPE>(helper, info)
    ESIG_MAKE_CTYPE_SWITCH(helper.ctype)
#undef ESIG_SWITCH_FN
}



esig::algebra::py_vector_construction_helper
esig::algebra::get_construction_data(const pybind11::object &arg,
                                     const pybind11::kwargs &kwargs) {

    auto helper = esig::algebra::kwargs_to_construction_data(kwargs);

    if (py::isinstance<py::buffer>(arg)) {
        auto info = arg.cast<py::buffer>().request();

        if (info.ndim != 1) {
            throw py::value_error("data has invalid shape");
        }

        coefficient_type input_ctype;
        switch (info.format[0]) {
            case 'f':
                input_ctype = coefficient_type::sp_real;
                break;
            case 'd':
                input_ctype = coefficient_type::dp_real;
                break;
            default:
                throw py::type_error("unsupported data type");
        }

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
