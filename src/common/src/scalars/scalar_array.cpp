//
// Created by sam on 10/11/22.
//

#include "esig/scalars.h"

using namespace esig;
using namespace scalars;

scalar_array::scalar_array(const scalar_array &other) noexcept
    : scalar_pointer(other), m_size(other.m_size)
{
}
scalar_array::scalar_array(scalar_array &&other) noexcept
    : scalar_pointer(other), m_size(other.m_size)
{
    other.p_data = nullptr;
    other.p_type = nullptr;
    other.m_size = 0;
}

scalar_array::scalar_array(const scalar_type* type)
    : scalar_pointer(type), m_size(0)
{}

scalar_array::scalar_array(void *data, const scalar_type *type, dimn_t size)
    : scalar_pointer(data, type), m_size(size)
{
}
scalar_array::scalar_array(const void *data, const scalar_type *type, dimn_t size)
    : scalar_pointer(data, type), m_size(size)
{
}

scalar_array &scalar_array::operator=(const scalar_array &other) noexcept {
    if (&other != this) {
        scalar_pointer::operator=(other);
        m_size = other.m_size;
    }
    return *this;
}
scalar_array &scalar_array::operator=(scalar_array &&other) noexcept {
    if (&other != this) {
        scalar_pointer::operator=(other);
        m_size = other.m_size;
        other.p_data = nullptr;
        other.p_type = nullptr;
        other.m_size = 0;
    }
    return *this;
}
