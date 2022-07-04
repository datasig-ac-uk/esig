//
// Created by user on 30/03/2022.
//

#include <esig/paths/path.h>
#include <cmath>

namespace esig {
namespace paths {


path_interface::path_interface(path_metadata metadata)
    : m_metadata(metadata)
{
}
const path_metadata &path_interface::metadata() const noexcept
{
    return m_metadata;
}
path_interface::compute_depth_t
path_interface::compute_depth(path_interface::accuracy_t accuracy) const noexcept
{
    int exponent;
    frexp(accuracy, &exponent);
    return static_cast<compute_depth_t>(-std::min(-0, exponent-1));
}
bool path_interface::empty(const interval &domain) const
{
    return true;
}
algebra::free_tensor
path_interface::signature(const interval &domain, path_interface::compute_depth_t resolution, const algebra::context &ctx) const
{
    return ctx.to_signature(log_signature(domain, resolution, ctx));
}
algebra::lie path_interface::log_signature(const dyadic_interval &domain, path_interface::compute_depth_t resolution, const algebra::context &ctx) const
{
    return log_signature(domain, ctx);
}
algebra::lie path_interface::log_signature(const interval &domain, path_interface::compute_depth_t resolution, const algebra::context &ctx) const
{
    std::cerr << "logsig on " << domain << '\n';

    // TODO: Replace with proper invocations
    return log_signature(domain, ctx);
}

std::shared_ptr<algebra::context> path::get_default_context() const
{
    auto metadata = p_impl->metadata();
    return algebra::get_context(metadata.width, metadata.depth, metadata.ctype, {});
}
algebra::lie path::log_signature(const interval &domain, path::accuracy_t accuracy) const
{
    std::cout << "log signature in path\n";
    auto ctx = get_default_context();
    auto resolution = p_impl->compute_depth(accuracy);
    return p_impl->log_signature(domain, resolution, *ctx);
}
algebra::lie path::log_signature(const interval &domain, path::accuracy_t accuracy, const algebra::context &ctx) const
{
    auto resolution = p_impl->compute_depth(accuracy);
    return p_impl->log_signature(domain, resolution, ctx);
}
algebra::free_tensor path::signature(const interval &domain, path::accuracy_t accuracy) const
{
    auto ctx = get_default_context();
    auto resolution = p_impl->compute_depth(accuracy);
    return p_impl->signature(domain, resolution, *ctx);
}
algebra::free_tensor path::signature(const interval &domain, path::accuracy_t accuracy, const algebra::context &ctx) const
{
    auto resolution = p_impl->compute_depth(accuracy);
    return p_impl->signature(domain, resolution, ctx);
}
algebra::free_tensor path::signature_derivative(const interval &domain, const algebra::lie &perturbation, path::accuracy_t accuracy) const
{
    auto ctx = get_default_context();
    auto metadata = p_impl->metadata();
    algebra::derivative_compute_info info{
         log_signature(domain, accuracy, *ctx),
         perturbation
    };
    return ctx->sig_derivative({info},
                               metadata.result_vec_type,
                               metadata.input_vec_type
                               );
}
algebra::free_tensor path::signature_derivative(const interval &domain, const algebra::lie &perturbation, path::accuracy_t accuracy, const algebra::context &ctx) const
{
    auto metadata = p_impl->metadata();
    algebra::derivative_compute_info info {
        log_signature(domain, accuracy, ctx),
        perturbation
    };
    return ctx.sig_derivative({info},
                              metadata.result_vec_type,
                              metadata.input_vec_type
                              );
}
algebra::free_tensor path::signature_derivative(const path::perturbation_list_t &perturbations, path::accuracy_t accuracy) const
{
    auto ctx = get_default_context();
    return signature_derivative(perturbations, accuracy, *ctx);
}
algebra::free_tensor path::signature_derivative(const path::perturbation_list_t &perturbations, path::accuracy_t accuracy, const algebra::context &ctx) const
{
    auto metadata = p_impl->metadata();
    auto resolution = p_impl->compute_depth(accuracy);

    std::vector<algebra::derivative_compute_info> info;
    info.reserve(perturbations.size());

    for (auto& val : perturbations) {
        info.push_back({p_impl->log_signature(val.first, resolution, ctx),
                        val.second});
    }

    return ctx.sig_derivative(info,
                              metadata.result_vec_type,
                              metadata.input_vec_type
                              );
}
const path_interface &path_base_access::get(const path &p) noexcept
{
    return *p.p_impl;
}



}// namespace paths
}// namespace esig
