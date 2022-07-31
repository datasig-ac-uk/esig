//
// Created by user on 28/07/22.
//

#ifndef ESIG_SRC_PATHS_SRC_PY_PATHS_PYTHON_ARGUMENTS_H_
#define ESIG_SRC_PATHS_SRC_PY_PATHS_PYTHON_ARGUMENTS_H_

#include <esig/algebra/context.h>
#include <esig/paths/path.h>

#include <pybind11/pybind11.h>

namespace esig {
namespace paths {



template<typename T>
void copy_C_array(const char *src,
                  char *dst,
                  const idimn_t ndim,
                  const idimn_t *shape) noexcept {
    idimn_t size = 1;
    for (idimn_t i = 0; i < ndim; ++i) {
        size *= shape[i];
    }
    const auto *begin = reinterpret_cast<const T *>(src);
    const auto *end = begin + size;
    auto *dest_p = reinterpret_cast<T *>(dst);
    std::uninitialized_copy(begin, end, dest_p);
}




inline bool is_C_array(const idimn_t itemsize, const idimn_t ndim, const idimn_t *shape, const idimn_t *strides) noexcept {
    auto stride = itemsize;
    for (idimn_t i = ndim; i >= 1;) {
        if (strides[--i] != stride) {
            return false;
        }
        stride *= shape[i];
    }
    return true;
}

template<typename It, typename Ot>
void copy_strided_array(const char *src,
                        const char *dst,
                        const idimn_t ndim,
                        const idimn_t *shape,
                        const idimn_t *strides) noexcept {
    const auto stride = *strides;
    const auto size = *shape;
    if (ndim == 1) {
        for (idimn_t i = 0; i < size; ++i) {
            ::new ((void *)dst) Ot(*((const It *)src));
            src += stride;
            dst += sizeof(Ot);
        }
    } else {
        dimn_t ostride = sizeof(Ot);
        for (idimn_t i = 1; i < ndim; ++i) {
            ostride *= shape[i];
        }

        for (idimn_t i = 0; i < size; ++i) {
            copy_strided_array<It, Ot>(src, dst, ndim - 1, shape + 1, strides + 1);
            src += stride;
            dst += ostride;
        }
    }
}

template <typename T>
void copy_cast_array(const char* src, char* dst, const idimn_t ndim, const idimn_t* shape, const idimn_t* strides)
{
    const auto stride = *strides;
    const auto size = *shape;
    if (ndim == 1) {
        for (idimn_t i = 0; i < size; ++i) {
            auto h = pybind11::reinterpret_borrow<pybind11::object>(*((PyObject**) src));
            ::new ((void *)dst) T(h.cast<T>());
            src += stride;
            dst += sizeof(T);
        }
    } else {
        dimn_t ostride = sizeof(T);
        for (idimn_t i = 1; i < ndim; ++i) {
            ostride *= shape[i];
        }

        for (idimn_t i = 0; i < size; ++i) {
            copy_cast_array<T>(src, dst, ndim-1, shape + 1, strides + 1);
            src += stride;
            dst += ostride;
        }
    }
}

template <typename It, typename Ot>
void copy_array(const void* src, char* dst, const idimn_t ndim, const idimn_t* shape, const idimn_t* strides)
{
    // The compiler should delete the branch that does not apply
    // when the expression is known at compile time. We're keeping
    // the inner branching separate because it is runtime.
    if (std::is_same<It, Ot>::value) {
        if (is_C_array(sizeof(Ot), ndim, shape, strides)) {
            // If we have a two C-arrays (contiguous) data of the same
            // type then we can just use the standard C++ mechanisms
            // for copying the buffer.
            copy_C_array<Ot>((const char*) src, dst, ndim, shape);
        } else {
            // Otherwise there is a not-trivial stride to consider
            copy_strided_array<It, Ot>((const char*) src, dst, ndim, shape, strides);
        }
    } else if (std::is_same<It, PyObject*>::value) {
        // If we have an array of PyObject pointers then we have to do some
        // non-trivial conversions inside the copy operation. For this, we
        // call a different function. Hopefully this is never used!
        copy_cast_array<Ot>((const char*) src, dst, ndim, shape, strides);
    } else {
        copy_strided_array<It, Ot>((const char*) src, dst, ndim, shape, strides);
    }
}









template <algebra::coefficient_type CTypeIn, algebra::coefficient_type CTypeOut>
void copy_convert_matrix(const void* src, void* dst, dimn_t n_rows, dimn_t n_cols, const std::vector<dimn_t>& strides)
{
    using in_type = algebra::type_of_coeff<CTypeIn>;
    using out_type = algebra::type_of_coeff<CTypeOut>;

    const auto* in_p = reinterpret_cast<const in_type*>(src);
    auto* out_p = reinterpret_cast<out_type *>(dst);

    const auto istride = strides[1] / sizeof(in_type);
    const auto jstride = strides[0] / sizeof(in_type);
    for (dimn_t i=0; i<n_rows; ++i) {
        for (dimn_t j=0; j<n_cols; ++j) {
            // probably makes no difference at the moment, but using a
            // placement new might be better in the future if we have
            // dynamic types here.
            ::new (out_p+j) out_type(in_p[j*jstride]);
        }
        out_p += n_cols;
        in_p += istride;
    }
}

#define _COPY_ARRAY_INVOCATION(TYPE)  \
    copy_array<TYPE, out_type>(info.ptr, \
                               buffer, \
                               info.ndim,\
                               info.shape.data(), \
                               info.strides.data())

template<algebra::coefficient_type CType>
void copy_py_buffer_to_data_buffer(char* buffer, const pybind11::buffer_info &info) {
    using out_type = algebra::type_of_coeff<CType>;

    switch (info.format[0]) {
        case 'd':
            _COPY_ARRAY_INVOCATION(double);
            break;
        case 'f':
            _COPY_ARRAY_INVOCATION(float);
            break;
        case 'q':
        case 'Q':
            _COPY_ARRAY_INVOCATION(long long);
            break;
        case 'i':
        case 'I':
            _COPY_ARRAY_INVOCATION(int);
            break;
        case 'l':
        case 'L':
            _COPY_ARRAY_INVOCATION(long);
            break;
        case 'n':
        case 'N':
            _COPY_ARRAY_INVOCATION(pybind11::ssize_t);
            break;
        case 'h':
        case 'H':
            _COPY_ARRAY_INVOCATION(short);
            break;
        case 'b':
        case 'B':
        case 'c':
            _COPY_ARRAY_INVOCATION(char);
            break;
        default:
            throw pybind11::type_error("unsupported data type");
    }
}
#undef _COPY_ARRAY_INVOCATION

struct additional_args {
    deg_t width = 0;
    deg_t depth = 0;
    real_interval domain {0, 1};
    algebra::coefficient_type ctype = algebra::coefficient_type::dp_real;
    algebra::input_data_type data_type = algebra::input_data_type::value_array;
    algebra::vector_type result_vtype = algebra::vector_type::dense;
};


path_metadata parse_kwargs_to_metadata(const pybind11::kwargs& kwargs, additional_args args);



} // namespace paths
} // namespace esig

#endif//ESIG_SRC_PATHS_SRC_PY_PATHS_PYTHON_ARGUMENTS_H_
