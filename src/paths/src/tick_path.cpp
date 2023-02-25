//
// Created by user on 24/05/22.
//

#include "esig/paths/tick_path.h"

#include <cassert>
#include <algorithm>

namespace esig {
namespace paths {
tick_path::tick_path(
        path_metadata &&md,
        std::vector<std::pair<param_t, std::vector<key_type>>> &&index_data,
        scalars::owned_scalar_array &&data)
    : dyadic_caching_layer(std::move(md)),
      m_data(std::move(data)),
      m_index()
{
    m_index.reserve(index_data.size());

}

algebra::lie tick_path::log_signature(const interval &domain, const algebra::context &ctx) const
{
    const auto& md = metadata();

    algebra::signature_data data {
        scalars::scalar_stream(ctx.ctype()), {}, md.result_vec_type
    };

    auto begin = (domain.get_type() == interval_type::clopen)
                 ? m_index.lower_bound(domain.inf())
                 : m_index.upper_bound(domain.inf());

    auto end = (domain.get_type() == interval_type::clopen)
                 ? m_index.lower_bound(domain.sup())
                 : m_index.upper_bound(domain.sup());

    if (begin == end) {
        return ctx.zero_lie(md.result_vec_type);
    }

    data.data_stream.reserve_size(end - begin);

    for (auto it1 = begin, it = it1++; it1 != end; ++it, ++it1) {
        data.data_stream.push_back(m_data[it->second])
    }



    return ctx.log_signature(data);
}

}// namespace paths
}// namespace esig
