//
// Created by user on 05/04/2022.
//

#include "dynamically_generated_path.h"

namespace esig {
namespace paths {





bool dynamically_constructed_path::empty(const interval &domain) const
{
    return false;
}
algebra::lie dynamically_constructed_path::log_signature(const interval &domain, const algebra::context &ctx) const
{
    const auto& md = metadata();
    const auto buf = eval(domain);
    esig::algebra::signature_data data(
            md.ctype,
            md.result_vec_type,
            dtl::function_increment_iterator(buf.begin(), buf.end())
            );

    return ctx.log_signature(std::move(data));
}


dtl::function_increment_iterator::function_increment_iterator(const char *begin, const char *end)
    : m_begin(begin), m_end(end), state(true)
{
}
const char *dtl::function_increment_iterator::dense_begin()
{
    return m_begin;
}
const char *dtl::function_increment_iterator::dense_end()
{
    return m_end;
}
bool dtl::function_increment_iterator::next_sparse()
{
    return false;
}
const void *dtl::function_increment_iterator::sparse_kv_pair()
{
    return nullptr;
}
bool dtl::function_increment_iterator::advance()
{
    if (state) {
        state = false;
        return true;
    }
    return state;
}
bool dtl::function_increment_iterator::finished() const
{
    return state;
}
} // namespace paths
} // namespace esig
