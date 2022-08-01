//
// Created by sam on 09/05/22.
//
#include <esig/algebra/coefficients.h>


#include <cassert>
#include <utility>

namespace esig {
namespace algebra {

allocating_data_buffer::allocating_data_buffer(const allocating_data_buffer &other)
    : data_buffer(nullptr, nullptr, false), m_size(other.m_size), m_alloc(other.m_alloc)
{
    auto begin = other.data_begin;
    auto end = other.data_end;
    m_size = static_cast<size_type>(end - begin) / m_alloc->item_size();
//    assert(begin != nullptr && end != nullptr && begin != end);
//    assert(m_size > 0);
    if (m_size > 0) {
        data_begin = m_alloc->allocate(m_size);
        if (data_begin != nullptr) {
            data_end = data_begin + static_cast<size_type>(end - begin);
            std::uninitialized_copy(begin, end, data_begin);
        } else {
            throw std::runtime_error("unable to allocate");
        }
    }

}

allocating_data_buffer::allocating_data_buffer(allocating_data_buffer &&other) noexcept
    : data_buffer(other.data_begin, other.data_end, other.is_const),
      m_alloc(std::move(other.m_alloc)), m_size(other.m_size)
{
    other.data_begin = nullptr;
    other.data_end = nullptr;
}

allocating_data_buffer::allocating_data_buffer(allocating_data_buffer::allocator_type alloc, data_buffer::size_type n)
    : data_buffer(nullptr, nullptr, false), m_alloc(std::move(alloc)), m_size(n)
{
    if (m_size > 0) {
        data_begin = m_alloc->allocate(n);
    }
    if (data_begin != nullptr) {
        data_end = data_begin + n*m_alloc->item_size();
    }
}
allocating_data_buffer::allocating_data_buffer(allocating_data_buffer::allocator_type alloc, const char *begin, const char *end)
    : data_buffer(nullptr, nullptr, false), m_alloc(std::move(alloc))
{
    m_size = static_cast<size_type>(end - begin) / m_alloc->item_size();
//    assert(begin != nullptr && end != nullptr && begin != end);
//    assert(m_size > 0);
    data_begin = m_alloc->allocate(m_size);
    if (data_begin != nullptr) {
        data_end = data_begin + static_cast<size_type>(end - begin);
        std::uninitialized_copy(begin, end, data_begin);
    }
}
allocating_data_buffer::~allocating_data_buffer()
{
    if (data_begin != nullptr) {
        assert(m_size*m_alloc->item_size() == size());
        m_alloc->deallocate(data_begin, m_size);
    }
    data_begin = nullptr;
    data_end = nullptr;
}

typename data_buffer::size_type allocating_data_buffer::item_size() const noexcept
{
    return m_alloc->item_size();
}

void allocating_data_buffer::set_allocator_and_alloc(std::shared_ptr<data_allocator> &&alloc, dimn_t size) {
    if (m_alloc && data_begin != nullptr) {
        m_alloc->deallocate(data_begin, m_size);
    }
    m_alloc = std::move(alloc);
    m_size = size;
    data_begin = m_alloc->allocate(m_size);
    if (data_begin != nullptr) {
        data_end = data_begin + m_size*m_alloc->item_size();
    }
}

void allocating_data_buffer::swap(allocating_data_buffer &other) noexcept {
    std::swap(m_alloc, other.m_alloc);
    std::swap(m_size, other.m_size);
    data_buffer::swap(other);
}

allocating_data_buffer &allocating_data_buffer::operator=(const allocating_data_buffer &other) {
    this->allocating_data_buffer::~allocating_data_buffer();
    m_alloc = other.m_alloc;
    m_size = other.m_size;

    data_begin = m_alloc->allocate(m_size);
    if (data_begin != nullptr) {
        data_end = data_begin + m_size;
        std::uninitialized_copy(other.data_begin, other.data_end, data_begin);
    }
    return *this;
}
allocating_data_buffer &allocating_data_buffer::operator=(allocating_data_buffer &&other) noexcept {
    this->allocating_data_buffer::~allocating_data_buffer();
    m_alloc = std::move(other.m_alloc);
    m_size = other.m_size;
    data_buffer::operator=(std::move(other));
    return *this;
}

}// namespace algebra
}// namespace esig
