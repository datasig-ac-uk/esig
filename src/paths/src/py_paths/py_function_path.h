//
// Created by sam on 03/05/22.
//

#ifndef ESIG_PY_FUNCTION_PATH_H
#define ESIG_PY_FUNCTION_PATH_H

#include "esig/paths/python_interface.h"
#include "py_path.h"
#include <pybind11/numpy.h>

namespace esig {
namespace paths {

class py_dense_function_path : public dynamically_constructed_path
{
public:
    using func_type = std::function<pybind11::buffer(real_interval)>;
private:
    func_type m_func;
    std::shared_ptr<algebra::data_allocator> m_alloc;

public:

    py_dense_function_path(path_metadata md, func_type&& fn);

    algebra::allocating_data_buffer eval(const interval &domain) const override;
};

class py_sparse_function_path : public dynamically_constructed_path
{
    using func_type = std::function<pybind11::dict(real_interval)>;
    func_type m_func;
    std::shared_ptr<algebra::data_allocator> m_alloc;
public:

    py_sparse_function_path(
            path_metadata md,
            func_type&& fn
            );
    algebra::allocating_data_buffer eval(const interval& domain) const override;
};


void init_py_function_path(pybind11::module_& m);


}// namespace paths
}// namespace esig

#endif//ESIG_PY_FUNCTION_PATH_H
