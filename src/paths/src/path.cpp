//
// Created by user on 30/03/2022.
//

#include <esig/paths/path.h>
#include <cmath>
#include "esig/paths/piecewise_lie_path.h"

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
    return domain.inf() == domain.sup();
}
algebra::FreeTensor
path_interface::signature(const interval &domain, path_interface::compute_depth_t resolution, const algebra::context &ctx) const
{
    return ctx.to_signature(log_signature(domain, resolution, ctx));
}
algebra::Lie path_interface::log_signature(const dyadic_interval &domain, path_interface::compute_depth_t resolution, const algebra::context &ctx) const
{
    return log_signature(domain, ctx);
}
algebra::Lie path_interface::log_signature(const interval &domain, path_interface::compute_depth_t resolution, const algebra::context &ctx) const
{

    // TODO: Replace with proper invocations
    return log_signature(domain, ctx);
}

std::shared_ptr<const algebra::context> path::get_default_context() const
{
    return p_impl->metadata().ctx;
}
algebra::Lie path::log_signature(const interval &domain, path::accuracy_t accuracy) const
{
    auto ctx = get_default_context();
    auto resolution = p_impl->compute_depth(accuracy);
    return p_impl->log_signature(domain, resolution, *ctx);
}
algebra::Lie path::log_signature(const interval &domain, path::accuracy_t accuracy, const algebra::context &ctx) const
{
    auto resolution = p_impl->compute_depth(accuracy);
    return p_impl->log_signature(domain, resolution, ctx);
}

algebra::Lie path::log_signature(accuracy_t accuracy) const {
    return path::log_signature(accuracy, *get_default_context());
}
algebra::Lie path::log_signature(accuracy_t accuracy, const algebra::context &ctx) const {
    auto resolution = p_impl->compute_depth(accuracy);
    auto domain = p_impl->metadata().effective_domain;
    return p_impl->log_signature(domain, resolution, ctx);
}

algebra::FreeTensor path::signature(const interval &domain, path::accuracy_t accuracy) const
{
    auto ctx = get_default_context();
    auto resolution = p_impl->compute_depth(accuracy);
    return p_impl->signature(domain, resolution, *ctx);
}
algebra::FreeTensor path::signature(const interval &domain, path::accuracy_t accuracy, const algebra::context &ctx) const
{
    auto resolution = p_impl->compute_depth(accuracy);
    return p_impl->signature(domain, resolution, ctx);
}
algebra::FreeTensor path::signature(accuracy_t accuracy) const {
    return path::signature(accuracy, *get_default_context());
}
algebra::FreeTensor path::signature(accuracy_t accuracy, const algebra::context &ctx) const {
    auto resolution = p_impl->compute_depth(accuracy);
    auto domain = p_impl->metadata().effective_domain;
    return p_impl->signature(domain, resolution, ctx);
}

algebra::FreeTensor path::signature_derivative(const interval &domain, const algebra::Lie &perturbation, path::accuracy_t accuracy) const
{
    auto ctx = get_default_context();
    auto metadata = p_impl->metadata();
    algebra::derivative_compute_info info{
         log_signature(domain, accuracy, *ctx),
         perturbation
    };
    return ctx->sig_derivative({info},
                               metadata.result_vec_type,
                               metadata.result_vec_type
                               );
}
algebra::FreeTensor path::signature_derivative(const interval &domain, const algebra::Lie &perturbation, path::accuracy_t accuracy, const algebra::context &ctx) const
{
    auto metadata = p_impl->metadata();
    algebra::derivative_compute_info info {
        log_signature(domain, accuracy, ctx),
        perturbation
    };
    return ctx.sig_derivative({info},
                              metadata.result_vec_type,
                              metadata.result_vec_type
                              );
}
algebra::FreeTensor path::signature_derivative(const path::perturbation_list_t &perturbations, path::accuracy_t accuracy) const
{
    auto ctx = get_default_context();
    return signature_derivative(perturbations, accuracy, *ctx);
}
algebra::FreeTensor path::signature_derivative(const path::perturbation_list_t &perturbations, path::accuracy_t accuracy, const algebra::context &ctx) const
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
                              metadata.result_vec_type
                              );
}
const path_interface &path_base_access::get(const path &p) noexcept
{
    return *p.p_impl;
}

const path_metadata& path::metadata() const noexcept {
    return p_impl->metadata();
}

std::unique_ptr<const path_interface> path_base_access::take(path &&p) noexcept {
    return std::move(p.p_impl);
}

path path::simplify_path(const partition &part, accuracy_t accuracy) const {

    auto md = metadata();

    std::vector<typename piecewise_lie_path::lie_piece> parts;
    parts.reserve(part.size());

    for (dimn_t i=0; i<part.size(); ++i) {
        auto ivl = part[i];
        parts.emplace_back(ivl, log_signature(ivl, accuracy));
    }

    return path(piecewise_lie_path(std::move(parts), md));
}

}// namespace paths
}// namespace esig
