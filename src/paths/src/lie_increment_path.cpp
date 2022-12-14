//
// Created by user on 31/03/2022.
//

#include "lie_increment_path.h"

namespace esig {
namespace paths {


lie_increment_path::lie_increment_path(
        scalars::owned_scalar_array&& buffer,
        const std::vector<param_t>& indices,
        esig::paths::path_metadata metadata)
    : dyadic_caching_layer(std::move(metadata)), m_buffer(std::move(buffer)), m_data()
{
    const auto& md = this->metadata();
    for (dimn_t i=0; i<indices.size(); ++i) {
        m_data[indices[i]] = i*md.width;
    }
}


algebra::lie lie_increment_path::log_signature(const esig::interval &domain, const esig::algebra::context &ctx) const
{
    const auto& md = metadata();
    if (empty(domain)) {
        return ctx.zero_lie(md.result_vec_type);
    }

    esig::algebra::signature_data data {
        scalars::scalar_stream(ctx.ctype()), {}, md.result_vec_type
    };


    if (m_keys.empty()) {
        data.data_stream.set_elts_per_row(m_data.size() == 1
                                              ? m_buffer.size()
                                              : m_data.begin()[1].second - m_data.begin()[0].second);
    }


    auto begin = (domain.get_type() == interval_type::clopen)
        ? m_data.lower_bound(domain.inf())
        : m_data.upper_bound(domain.inf());

    auto end = (domain.get_type() == interval_type::clopen)
                 ? m_data.lower_bound(domain.sup())
                 : m_data.upper_bound(domain.sup());

    if (begin == end) {
        return ctx.zero_lie(md.result_vec_type);
    }

    data.data_stream.reserve_size(end - begin);


    for (auto it1 = begin, it = it1++; it1 != end; ++it, ++it1) {
        data.data_stream.push_back({m_buffer[it->second].to_const_pointer(), it1->second - it->second});
    }
    // Case it = it1 - 1 and it1 == end
    --end;
    data.data_stream.push_back({m_buffer[end->second].to_const_pointer(), m_buffer.size() - end->second});

    if (m_keys.empty()) {
        data.key_stream.reserve(end - begin);
        ++end;
        for (auto it=begin; it!= end; ++it) {
            data.key_stream.push_back(m_keys.data() + it->second);
        }
    }

    assert(ctx.width() == md.width);
//    assert(ctx.depth() == md.depth);

    return ctx.log_signature(std::move(data));
}
bool lie_increment_path::empty(const interval &domain) const
{
    auto current = m_data.lower_bound(domain.inf());
    auto end = m_data.lower_bound(domain.sup());
    return end == current;
}



}// namespace paths
}// namespace esig
