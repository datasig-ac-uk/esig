//
// Created by sam on 10/11/22.
//


#include "esig/scalars.h"


using namespace esig;
using namespace esig::scalars;

scalar_pointer::scalar_pointer(const void* data, const scalar_type* type, constness cness)
    : p_data(data), p_type(type), m_constness(cness)
{
    if (p_type == nullptr && p_data != nullptr) {
        throw std::runtime_error("non-zero values must have a type");
    }
}

scalar scalar_pointer::deref() {
    return scalar(*this);
}
scalar scalar_pointer::deref_mut() {
    if (m_constness == IsConst) {
        throw std::runtime_error("attempting to dereference const pointer to non-const value");
    }
    return scalar(*this);
}

scalar scalar_pointer::operator*() {
    return deref_mut();
}
scalar scalar_pointer::operator*() const noexcept {
    return scalar(*this);
}
scalar_pointer scalar_pointer::operator+(dimn_t index) const noexcept {
    if (p_data == nullptr) {
        return scalar_pointer(p_type);
    }
    return {static_cast<const char*>(p_data) + index*p_type->itemsize(), p_type, m_constness};
}
scalar_pointer &scalar_pointer::operator+=(dimn_t index) noexcept {
    if (p_data != nullptr) {
        p_data = static_cast<const char*>(p_data) + index*p_type->itemsize();
    }
    return *this;
}
scalar scalar_pointer::operator[](dimn_t index) {
    if (m_constness == IsConst) {
        throw std::runtime_error("attempting to dereference const pointer to non-const value");
    }
    return scalar((*this) + index);
}
scalar scalar_pointer::operator[](dimn_t index) const noexcept {
    return scalar((*this) + index);
}
