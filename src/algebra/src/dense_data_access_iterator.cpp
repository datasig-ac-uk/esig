//
// Created by user on 27/06/22.
//

#include <esig/algebra/iteration.h>


namespace esig {
namespace algebra {

dense_data_access_item::dense_data_access_item(key_type start_key, const void *begin, const void *end)
    : m_start_key(start_key), m_begin(begin), m_end(end)
{}


} // namespace algebra
} // namespace esig
