//
// Created by sam on 02/05/22.
//

#ifndef ESIG_PY_PATH_H
#define ESIG_PY_PATH_H

#include "esig/paths/path.h"
#include "esig/paths/python_interface.h"


namespace esig {
namespace paths {

path py_path_constructor(const pybind11::args &args, const pybind11::kwargs& kwargs);


struct py_path_interface : public path_interface {
    using path_interface::path_interface;
    compute_depth_t compute_depth(accuracy_t accuracy) const noexcept override;
    bool empty(const interval &domain) const override;
    algebra::lie log_signature(const interval &domain, const algebra::context &ctx) const override;
    algebra::lie log_signature(const dyadic_interval &domain, compute_depth_t resolution, const algebra::context &ctx) const override;
    algebra::lie log_signature(const interval &domain, compute_depth_t resolution, const algebra::context &ctx) const override;
    algebra::free_tensor signature(const interval &domain, compute_depth_t resolution, const algebra::context &ctx) const override;
};


class py_path_interface_wrapper : public path_interface
{
    pybind11::object obj;
public:

    explicit py_path_interface_wrapper(pybind11::object wrap);

    using path_interface::log_signature;
    algebra::lie log_signature(const interval &domain, const algebra::context &ctx) const override;
};

void init_python_path_interface(pybind11::module_ &m);


} // namespace paths
} // namespace esig



#endif//ESIG_PY_PATH_H
