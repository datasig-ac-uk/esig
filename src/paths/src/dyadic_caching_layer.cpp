//
// Created by user on 04/05/22.
//


#include <esig/paths/path.h>

esig::paths::dyadic_caching_layer::dyadic_caching_layer(esig::paths::dyadic_caching_layer &&other) noexcept
    : path_interface(other.metadata())
{
    std::lock_guard<std::recursive_mutex> access(other.m_compute_lock);
    m_cache.insert(other.m_cache.begin(), other.m_cache.end());
}
esig::paths::dyadic_caching_layer &esig::paths::dyadic_caching_layer::operator=(esig::paths::dyadic_caching_layer &&other) noexcept
{
    std::lock_guard<std::recursive_mutex> thisaccess(m_compute_lock), thataccess(other.m_compute_lock);
    m_cache.clear();
    path_interface::operator=(other);
    m_cache.insert(other.m_cache.begin(), other.m_cache.end());
    other.~dyadic_caching_layer();
    return *this;
}

esig::algebra::lie
esig::paths::dyadic_caching_layer::log_signature(
        const esig::dyadic_interval &domain,
        typename esig::paths::path_interface::compute_depth_t resolution,
        const algebra::context &ctx) const
{
    if (empty(domain)) {
        return ctx.zero_lie(metadata().result_vec_type);
    }

    if (domain.power() == resolution) {
        std::lock_guard<std::recursive_mutex> access(m_compute_lock);

        auto& cached = m_cache[domain];
        if (!cached) {
            cached = log_signature(domain, ctx);
        }
        return cached;
    }

    dyadic_interval lhs_itvl(domain), rhs_itvl(domain);
    lhs_itvl.shrink_interval_left();
    rhs_itvl.shrink_interval_right();

    auto lhs = log_signature(lhs_itvl, resolution, ctx);
    auto rhs = log_signature(rhs_itvl, resolution, ctx);

    return ctx.cbh({lhs, rhs}, metadata().result_vec_type);
}

esig::algebra::lie
esig::paths::dyadic_caching_layer::log_signature(
    const esig::interval &domain,
    esig::paths::path_interface::compute_depth_t resolution,
    const esig::algebra::context &ctx) const
{
    // For now, if the ctx depth is not the same as md depth just do the calculation without caching
    // be smarter about this in the future.
    const auto& md = metadata();
    assert(ctx.width() == md.width);
    if (ctx.depth() != md.depth) {
        return path_interface::log_signature(domain, resolution, ctx);
    }

    auto dyadic_dissection = esig::to_dyadic_intervals(domain.inf(), domain.sup(), resolution);

    std::vector<algebra::lie> lies;
    lies.reserve(dyadic_dissection.size());
    for (const auto& itvl : dyadic_dissection) {
        auto lsig = log_signature(itvl, resolution, ctx);
        lies.push_back(lsig);
    }

    return ctx.cbh(lies, metadata().result_vec_type);
}
