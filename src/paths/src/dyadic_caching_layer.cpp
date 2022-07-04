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
    std::lock_guard<std::recursive_mutex> access(m_compute_lock);

    std::cout << "Log signature interface\n";
    return ctx.zero_lie(algebra::vector_type::sparse);
}
