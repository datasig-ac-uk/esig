//
// Created by sam on 09/05/22.
//

#include <esig/algebra/coefficients.h>

#include <cassert>

namespace esig {
namespace algebra {


rowed_data_buffer::rowed_data_buffer(rowed_data_buffer::allocator_type alloc, data_buffer::size_type row_size, data_buffer::size_type nrows)
    : allocating_data_buffer(std::move(alloc), row_size*nrows), row_size(row_size)
{
}
rowed_data_buffer::rowed_data_buffer(rowed_data_buffer::allocator_type alloc, data_buffer::size_type row_size, const char *begin, const char *end)
    : allocating_data_buffer(std::move(alloc), begin, end), row_size(row_size)
{
}
rowed_data_buffer::range_type rowed_data_buffer::operator[](data_buffer::size_type rowid) const noexcept
{
    auto item_size = m_alloc->item_size();
    auto offset_begin = rowid*row_size*item_size;
    auto offset_end = (rowid+1)*row_size*item_size;
    assert(offset_begin < offset_end && offset_end <= size());
    return {begin() + offset_begin, begin() + offset_end};
}
rowed_data_buffer::rowed_data_buffer() : allocating_data_buffer(), row_size(0)
{}

}// namespace algebra
}// namespace esig
