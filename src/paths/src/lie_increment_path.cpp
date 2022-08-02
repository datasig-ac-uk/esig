//
// Created by user on 31/03/2022.
//

#include "lie_increment_path.h"

namespace esig {
namespace paths {

dense_increment_iterator::dense_increment_iterator(dense_increment_iterator::iterator_type b, dense_increment_iterator::iterator_type e)
    : m_current(b), m_end(e)
{
}
lie_increment_path::lie_increment_path(
        algebra::rowed_data_buffer&& buffer,
        const std::vector<param_t>& indices,
        esig::paths::path_metadata metadata)
    : dyadic_caching_layer(std::move(metadata)), m_buffer(buffer), m_data()
{
    assert(indices.empty() || buffer.begin() != nullptr);
    assert(indices.empty() || m_buffer.begin() != nullptr);
    for (dimn_t i=0; i<indices.size(); ++i) {
        m_data[indices[i]] = m_buffer[i];
    }
}


algebra::lie lie_increment_path::log_signature(const esig::interval &domain, const esig::algebra::context &ctx) const
{
    const auto& md = metadata();
    if (empty(domain)) {
        return ctx.zero_lie(md.result_vec_type);
    }



    esig::algebra::signature_data data (
            md.ctype,
            md.result_vec_type,
            dense_increment_iterator(
                    m_data.lower_bound(domain.inf()),
                    m_data.lower_bound(domain.sup())
                    )
            );

    assert(ctx.width() == md.width);
    assert(ctx.depth() == md.depth);

    return ctx.log_signature(std::move(data));
}
bool lie_increment_path::empty(const interval &domain) const
{
    auto current = m_data.lower_bound(domain.inf());
    auto end = m_data.lower_bound(domain.sup());
    return end == current;
}


const char *dense_increment_iterator::dense_begin()
{
    return m_current->second.first;
}
const char *dense_increment_iterator::dense_end()
{
    return m_current->second.second;
}
bool dense_increment_iterator::next_sparse()
{
    return false;
}
const void *dense_increment_iterator::sparse_kv_pair()
{
    return nullptr;
}
bool dense_increment_iterator::advance()
{
    ++m_current;
    return m_current != m_end;
}
bool dense_increment_iterator::finished() const
{
    return m_current == m_end;
}

}// namespace paths
}// namespace esig
