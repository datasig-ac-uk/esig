//
// Created by user on 24/05/22.
//

#include "tick_path.h"

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
    const dimn_t item_size = m_data.item_size();
    const char* position = m_data.begin();

    std::sort(index_data.begin(), index_data.end(),
              [](const std::pair<key_type, dimn_t>& i1,
                 const std::pair<key_type, dimn_t>& i2) {
                  return i1.first < i2.first;
              });

    for (auto item : index_data) {
        m_index[item.first] = {position, item.second};
        position += item.second*item_size;
        assert(position <= m_data.end());
    }
    assert(position == m_data.end());
}

algebra::lie tick_path::log_signature(const interval &domain, const algebra::context &ctx) const
{
    const auto& md = metadata();

    algebra::signature_data data {
            md.ctype,
            md.result_vec_type,
            tick_increment_iterator {
                    m_index.lower_bound(domain.inf()),
                    m_index.lower_bound(domain.sup()),
                    m_data.item_size()
            }
    };
    return ctx.log_signature(std::move(data));
}

}// namespace paths
}// namespace esig
