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
        std::vector<std::pair<param_t, dimn_t>> &&index_data,
        algebra::allocating_data_buffer &&data)
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
            md.input_vec_type,
            tick_increment_iterator {
                    m_index.lower_bound(domain.inf()),
                    m_index.lower_bound(domain.sup()),
                    m_data.item_size()
            }
    };
    return ctx.log_signature(std::move(data));
}

tick_increment_iterator::tick_increment_iterator(
        tick_increment_iterator::iterator_type b,
        tick_increment_iterator::iterator_type e,
        dimn_t pair_size)
    : m_current(b), m_end(e), m_index(0), m_pair_size(pair_size)
{
}

const char *tick_increment_iterator::dense_begin()
{
    return nullptr;
}
const char *tick_increment_iterator::dense_end()
{
    return nullptr;
}
bool tick_increment_iterator::next_sparse()
{
    assert(m_current != m_end);
    return m_index < m_current->second.count;
}
const void *tick_increment_iterator::sparse_kv_pair()
{
    assert(m_current != m_end);
    assert(m_index < m_current->second.count);
    dimn_t idx = m_index++;
    return m_current->second.ptr + m_pair_size*idx;
}
bool tick_increment_iterator::advance()
{
    ++m_current;
    return m_current != m_end;
}
bool tick_increment_iterator::finished() const
{
    return m_current == m_end;
}
}// namespace paths
}// namespace esig
