//
// Created by sam on 10/11/22.
//

#include "esig/scalar_array.h"

using namespace esig;
using namespace scalars;

ScalarArray::ScalarArray(const ScalarArray &other) noexcept
    : ScalarPointer(other), m_size(other.m_size)
{
}
ScalarArray::ScalarArray(ScalarArray &&other) noexcept
    : ScalarPointer(other), m_size(other.m_size)
{
    other.p_data = nullptr;
    other.p_type = nullptr;
    other.m_size = 0;
}

ScalarArray::ScalarArray(const ScalarType * type)
    : ScalarPointer(type), m_size(0)
{}

ScalarArray::ScalarArray(void *data, const ScalarType *type, dimn_t size)
    : ScalarPointer(data, type), m_size(size)
{
}
ScalarArray::ScalarArray(const void *data, const ScalarType *type, dimn_t size)
    : ScalarPointer(data, type), m_size(size)
{
}

ScalarArray &ScalarArray::operator=(const ScalarArray &other) noexcept {
    if (&other != this) {
        ScalarPointer::operator=(other);
        m_size = other.m_size;
    }
    return *this;
}
ScalarArray &ScalarArray::operator=(ScalarArray &&other) noexcept {
    if (&other != this) {
        ScalarPointer::operator=(other);
        m_size = other.m_size;
        other.p_data = nullptr;
        other.p_type = nullptr;
        other.m_size = 0;
    }
    return *this;
}
