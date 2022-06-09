//
// Created by sam on 03/05/22.
//


#include "py_function_path.h"

#include <esig/algebra/coefficients.h>

#include <pybind11/functional.h>

#include <cassert>

namespace py = pybind11;

namespace esig {
namespace paths {

py_dense_function_path::py_dense_function_path(py_dense_function_path::func_type &&fn, deg_t width, deg_t depth, real_interval domain, algebra::coefficient_type ctype)
    : dynamically_constructed_path(path_metadata{
              width, depth, std::move(domain), ctype, algebra::vector_type::dense, algebra::vector_type::dense
      }),
      m_func(std::move(fn)), m_alloc(algebra::allocator_for_coeff(ctype))
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

py_sparse_function_path::py_sparse_function_path(py_sparse_function_path::func_type &&fn, deg_t width, deg_t depth, real_interval domain, algebra::coefficient_type ctype)
    : dynamically_constructed_path(path_metadata{width, depth, std::move(domain), algebra::coefficient_type::dp_real,
                                                 algebra::vector_type::sparse, algebra::vector_type::sparse}),
      m_func(std::move(fn)), m_alloc(algebra::allocator_for_key_coeff(algebra::coefficient_type::dp_real))
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


void init_py_function_path(pybind11::module_ &m)
{
    py::class_<py_dense_function_path, path_interface> dense_fdp(m, "DenseFunctionPath");
    py::class_<py_sparse_function_path, path_interface> sparse_fdp(m, "SparseFunctionPath");

    path_constructor func([](const py::args& args, const py::kwargs& kwargs) {
        return std::move(esig::paths::path(py_dense_function_path(
                args[0].cast<typename py_dense_function_path::func_type>(),
                kwargs["width"].cast<deg_t>(),
                kwargs["depth"].cast<deg_t>(),
                kwargs["domain"].cast<real_interval>(),
                kwargs["ctype"].cast<algebra::coefficient_type>()
                )));
    });
    esig::paths::register_pypath_constructor(py::type::of<py_dense_function_path>(), std::move(func));


}


}// namespace paths
}// namespace esig
