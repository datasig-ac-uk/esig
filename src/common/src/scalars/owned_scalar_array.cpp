//
// Created by user on 15/11/22.
//

#include "esig/owned_scalar_array.h"
#include "esig/scalar_pointer.h"
#include "esig/scalar_type.h"
#include "esig/scalar.h"
#include "esig/scalar_array.h"

using namespace esig;
using namespace scalars;

OwnedScalarArray::OwnedScalarArray(const ScalarType *type)
    : ScalarArray(type) {
}
OwnedScalarArray::OwnedScalarArray(const ScalarArray &other)
    : ScalarArray(other.type()->allocate(other.size()), other.size())
{
    if (other.ptr() != nullptr) {
        p_type->convert_copy(const_cast<void*>(p_data),
                             ScalarPointer(other),
                             other.size());
    }
}
OwnedScalarArray::OwnedScalarArray(const OwnedScalarArray &other)
    : ScalarArray(other.type()->allocate(other.size()), other.size())
{
    p_type->convert_copy(const_cast<void*>(p_data),
                         ScalarPointer(other),
                         other.size());
}

OwnedScalarArray::OwnedScalarArray(const ScalarType *type, dimn_t size)
    : ScalarArray(type->allocate(size), size)
{}

OwnedScalarArray::OwnedScalarArray(const Scalar &value, dimn_t count)
    : ScalarArray((value.type() != nullptr
                    ? value.type()->allocate(count)
                    : ScalarPointer()),
                   count)
{
    if (p_data != nullptr && p_type != nullptr) {
        p_type->assign(const_cast<void*>(p_data), value.to_const_pointer(), count);
    }
}

OwnedScalarArray::OwnedScalarArray(const ScalarType *type, const void *data, dimn_t count) noexcept
    : ScalarArray(data, type, count)
{
}

OwnedScalarArray::~OwnedScalarArray()
{
    if (p_data != nullptr) {
        OwnedScalarArray::p_type->deallocate(
            *this,
            OwnedScalarArray::m_size);
    }
}

OwnedScalarArray::OwnedScalarArray(OwnedScalarArray &&other) noexcept
    : ScalarArray(std::move(other))
{
}
OwnedScalarArray &OwnedScalarArray::operator=(const ScalarArray &other) {
    if (&other != this) {
        this->~OwnedScalarArray();
        if (other.size() > 0) {
            ScalarPointer::operator=(other.type()->allocate(other.size()));
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
OwnedScalarArray &OwnedScalarArray::operator=(OwnedScalarArray &&other) noexcept
{
    if (&other != this) {
        ScalarPointer::operator=(other);
        m_size = other.m_size;
        other.p_data = nullptr;
        other.p_type = nullptr;
        other.m_size = 0;
    }
    return *this;
}
