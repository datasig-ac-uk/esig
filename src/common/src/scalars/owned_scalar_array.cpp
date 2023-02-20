//
// Created by user on 15/11/22.
//

#include "esig/scalars.h"


using namespace esig;
using namespace scalars;

owned_scalar_array::owned_scalar_array(const scalar_type *type)
    : scalar_array(type) {
}
owned_scalar_array::owned_scalar_array(const scalar_array &other)
    : scalar_array(other.type()->allocate(other.size()), other.size())
{
    p_type->convert_copy(const_cast<void*>(p_data),
                         scalar_pointer(other),
                         other.size());
}
owned_scalar_array::owned_scalar_array(const owned_scalar_array &other)
    : scalar_array(other.type()->allocate(other.size()), other.size())
{
    p_type->convert_copy(const_cast<void*>(p_data),
                         scalar_pointer(other),
                         other.size());
}

owned_scalar_array::owned_scalar_array(const scalar_type *type, dimn_t size)
    : scalar_array(type->allocate(size), size)
{}

owned_scalar_array::owned_scalar_array(const scalar &value, dimn_t count)
    : scalar_array((value.type() != nullptr
                    ? value.type()->allocate(count)
                    : scalar_pointer()),
                   count)
{
    if (p_data != nullptr && p_type != nullptr) {
        p_type->assign(const_cast<void*>(p_data), value.to_const_pointer(), count);
    }
}

owned_scalar_array::owned_scalar_array(const scalar_type *type, const void *data, dimn_t count) noexcept
    : scalar_array(data, type, count)
{
}

owned_scalar_array::~owned_scalar_array()
{
    if (p_data != nullptr) {
        owned_scalar_array::p_type->deallocate(
            *this,
            owned_scalar_array::m_size);
    }
}

owned_scalar_array::owned_scalar_array(owned_scalar_array &&other) noexcept
    : scalar_array(std::move(other))
{
}
owned_scalar_array &owned_scalar_array::operator=(const scalar_array &other) {
    if (&other != this) {
        this->~owned_scalar_array();
        if (other.size() > 0) {
            scalar_pointer::operator=(other.type()->allocate(other.size()));
            m_size = other.size();
            p_type->convert_copy(const_cast<void*>(p_data), other, m_size);
        } else {
            p_data = nullptr;
            m_size = 0;
            p_type = other.type();
        }
    }
    return *this;
}
owned_scalar_array &owned_scalar_array::operator=(owned_scalar_array &&other) noexcept
{
    if (&other != this) {
        scalar_pointer::operator=(other);
        m_size = other.m_size;
        other.p_data = nullptr;
        other.p_type = nullptr;
        other.m_size = 0;
    }
    return *this;
}
