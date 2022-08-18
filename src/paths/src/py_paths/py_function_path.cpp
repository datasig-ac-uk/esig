//
// Created by sam on 03/05/22.
//


#include "py_function_path.h"

#include <esig/algebra/coefficients.h>
#include "python_arguments.h"

#include <pybind11/functional.h>

#include <cassert>
#include <iostream>

namespace py = pybind11;

namespace esig {
namespace paths {

namespace {

using algebra::coefficient_type;
template <typename S, typename T>
void compute_difference(char* out, const py::buffer_info& sinfo, const py::buffer_info& iinfo)
{
    const auto* lhs = reinterpret_cast<const T*>(sinfo.ptr);
    const auto* rhs = reinterpret_cast<const T*>(iinfo.ptr);
    auto* out_p = reinterpret_cast<S*>(out);

    for (idimn_t i=0; i<sinfo.size; ++i) {
        ::new (out_p + i) S(lhs[i] - rhs[i]);
    }
}



#define ESIG_COMPUTE_IMPL(TYPE)  compute_difference<scalar_type, TYPE>(result.begin(), sinfo, iinfo)

template <esig::algebra::coefficient_type CType>
algebra::allocating_data_buffer compute_dense_increment_impl(const py::buffer_info& sinfo, const py::buffer_info& iinfo)
{
    using scalar_type = esig::algebra::type_of_coeff<CType>;
    algebra::allocating_data_buffer result(algebra::allocator_for_coeff(CType), sinfo.size);

    switch (sinfo.format[0]) {
        case 'd':
            ESIG_COMPUTE_IMPL(double);
            break;
        case 'f':
            ESIG_COMPUTE_IMPL(float);
            break;
        case 'q':
        case 'Q':
            ESIG_COMPUTE_IMPL(long long);
            break;
        case 'i':
        case 'I':
            ESIG_COMPUTE_IMPL(int);
            break;
        case 'l':
        case 'L':
            ESIG_COMPUTE_IMPL(long);
            break;
        case 'n':
        case 'N':
            ESIG_COMPUTE_IMPL(py::ssize_t);
            break;
        case 'h':
        case 'H':
            ESIG_COMPUTE_IMPL(short);
            break;
        case 'b':
        case 'B':
        case 'c':
            ESIG_COMPUTE_IMPL(char);
            break;
        default:
            throw py::type_error("unsupported data type");
    }

    return result;
}

#undef ESIG_COMPUTE_IMPL

algebra::allocating_data_buffer compute_dense_increment(esig::algebra::coefficient_type ctype, py::buffer_info sinfo, py::buffer_info iinfo)
{
    assert(sinfo.size == iinfo.size);
    assert(sinfo.itemsize == iinfo.itemsize);
    assert(sinfo.format == iinfo.format);
    assert(sinfo.ndim == 1 && iinfo.ndim == 1);
    assert(sinfo.shape[0] == iinfo.shape[0]);

#define ESIG_SWITCH_FN(C) compute_dense_increment_impl<C>(sinfo, iinfo)
    ESIG_MAKE_CTYPE_SWITCH(ctype)
#undef ESIG_SWITCH_FN
}

}



py_dense_function_path::py_dense_function_path(path_metadata md, py_dense_function_path::func_type &&fn)
    : dynamically_constructed_path(std::move(md)),
      m_func(std::move(fn)), m_alloc(algebra::allocator_for_coeff(metadata().ctype))
{
}
algebra::allocating_data_buffer py_dense_function_path::eval(const interval &domain) const
{
    auto val_sup = m_func(domain.sup());
//    py::print("sup", val_sup);
    auto val_inf = m_func(domain.inf());
//    py::print("inf", val_inf);
    return compute_dense_increment(metadata().ctype, val_sup.request(), val_inf.request());
}

py_sparse_function_path::py_sparse_function_path(path_metadata md, py_sparse_function_path::func_type &&fn)
    : dynamically_constructed_path(std::move(md)),
      m_func(std::move(fn)),
      m_alloc(algebra::allocator_for_key_coeff(metadata().ctype))
{
}
algebra::allocating_data_buffer py_sparse_function_path::eval(const interval &domain) const
{
    auto val_sup = m_func(domain.sup());
    auto val_inf = m_func(domain.inf());
    algebra::allocating_data_buffer result(m_alloc, val_sup.size());
    char* ptr = result.begin();

    for (const auto& item : val_sup) {
        auto first = item.first.cast<key_type>();
        auto second = item.second.cast<scalar_t>();  // all python float values are doubles
        std::memcpy(ptr, &first, sizeof(key_type));
        std::memcpy(ptr + sizeof(key_type), &second, size_of(metadata().ctype));
        ptr += m_alloc->item_size();
        assert(ptr <= result.end());
    }
    assert(ptr == result.end());
    return result;
}


esig::paths::path construct_function_path(const py::args& args, const py::kwargs& kwargs)
{
    additional_args more_args;
    auto md = esig::paths::parse_kwargs_to_metadata(kwargs, more_args);


    return esig::paths::path(
        py_dense_function_path(md, args[0].cast<typename py_dense_function_path::func_type>())
        );
}


void init_py_function_path(pybind11::module_ &m)
{
    py::class_<py_dense_function_path, path_interface> dense_fdp(m, "DenseFunctionPath");
    py::class_<py_sparse_function_path, path_interface> sparse_fdp(m, "SparseFunctionPath");


    esig::paths::register_pypath_constructor(py::type::of<py_dense_function_path>(), &construct_function_path);


}


}// namespace paths
}// namespace esig
