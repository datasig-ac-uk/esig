//
// Created by user on 26/07/22.
//


#include <esig/algebra/python_interface.h>
#include "convert_buffer.h"


namespace py = pybind11;

using esig::dimn_t;
using esig::key_type;
using esig::algebra::coefficient_type;


template<coefficient_type CType>
void convert_buffer_impl(std::vector<char> &buffer, const py::buffer_info &info) {
    using scalar_type = esig::algebra::type_of_coeff<CType>;
    buffer.resize(sizeof(scalar_type) * info.size);

    using esig::algebra::copy_convert;

    switch (info.format[0]) {
        case 'f':
            copy_convert<scalar_type, float>(buffer.data(), info.ptr, info.size);
            break;
        case 'd':
            copy_convert<scalar_type, double>(buffer.data(), info.ptr, info.size);
            break;
    }
}

template <coefficient_type CType>
void to_kv_pairs_impl(std::vector<char>& buffer, const py::buffer_info& info) {
    using scalar_type = esig::algebra::type_of_coeff<CType>;

    using kv_t = std::pair<key_type, scalar_type>;

    const dimn_t n_keys = buffer.size() / sizeof(key_type);
    // Buffer is assumed to contain the keys that will be used, so the first task is to
    // make a copy that contains just the keys as key_types
    const auto* key_begin = reinterpret_cast<const key_type*>(buffer.data());
    const auto* key_end = key_begin + n_keys;
    std::vector<key_type> keys(key_begin, key_end);

    // Now increase the size of the vector to accommodate pairs rather than
    // just keys
    if (n_keys != info.size) {
        throw py::value_error("mismatching number of keys and values");
    }
    buffer.resize(n_keys);

    // copy over the key, values
    auto* outp = reinterpret_cast<kv_t*>(buffer.data());
    const auto* inp = reinterpret_cast<
    for (dimn_t i=0; i<n_keys; ++i) {
        outp[i] =
    }


}


void esig::algebra::convert_buffer(std::vector<char> &buffer, const py::buffer_info &info, coefficient_type ctype) {
#define ESIG_SWITCH_FN(CTYPE) convert_buffer_impl<CTYPE>(buffer, info)
    ESIG_MAKE_CTYPE_SWITCH(ctype)
#undef ESIG_SWITCH_FN
}


void to_kv_pairs(std::vector<char>& buffer, const py::buffer_info& info, coefficient_type ctype)
{

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


        if (helper.ctype_requested && input_ctype != helper.ctype) {
            convert_buffer(helper.buffer, info, helper.ctype);
            helper.begin_ptr = helper.buffer.data();
            helper.end_ptr = helper.begin_ptr + helper.buffer.size();
        } else {
            helper.ctype = input_ctype;
            helper.begin_ptr = reinterpret_cast<const char *>(info.ptr);
            helper.end_ptr = helper.begin_ptr + info.size * info.itemsize;
        }

    }

    return helper;
}
