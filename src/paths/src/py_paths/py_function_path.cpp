//
// Created by sam on 03/05/22.
//


#include "py_function_path.h"

#include <esig/algebra/coefficients.h>
#include "python_arguments.h"

#include <pybind11/functional.h>

#include <cassert>

namespace py = pybind11;

namespace esig {
namespace paths {

py_dense_function_path::py_dense_function_path(path_metadata md, py_dense_function_path::func_type &&fn)
    : dynamically_constructed_path(std::move(md)),
      m_func(std::move(fn)), m_alloc(algebra::allocator_for_coeff(metadata().ctype))
{
}
algebra::allocating_data_buffer py_dense_function_path::eval(const interval &domain) const
{
    auto val = m_func(real_interval(domain));
    auto buf_info = val.request();
    assert(buf_info.ndim == 1);
    assert(buf_info.itemsize == size_of(metadata().ctype));

    auto* ptr = reinterpret_cast<const char*>(buf_info.ptr);
    return {m_alloc, ptr, ptr+buf_info.size*buf_info.itemsize};
}

py_sparse_function_path::py_sparse_function_path(path_metadata md, py_sparse_function_path::func_type &&fn)
    : dynamically_constructed_path(std::move(md)),
      m_func(std::move(fn)),
      m_alloc(algebra::allocator_for_key_coeff(metadata().ctype))
{
}
algebra::allocating_data_buffer py_sparse_function_path::eval(const interval &domain) const
{
    auto val = m_func(real_interval(domain));
    algebra::allocating_data_buffer result(m_alloc, val.size());
    char* ptr = result.begin();

    for (const auto& item : val) {
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
