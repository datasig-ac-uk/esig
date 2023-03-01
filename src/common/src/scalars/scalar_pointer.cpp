//
// Created by sam on 10/11/22.
//


#include "esig/scalar_pointer.h"
#include "esig/scalar.h"

using namespace esig;
using namespace esig::scalars;

ScalarPointer::ScalarPointer(const void* data, const ScalarType * type, Constness cness)
    : p_data(data), p_type(type), m_constness(cness)
{
}

Scalar ScalarPointer::deref() {
    return Scalar(*this);
}
Scalar ScalarPointer::deref_mut() {
    if (m_constness == IsConst) {
        throw std::runtime_error("attempting to dereference const pointer to non-const value");
    }
    return Scalar(*this);
}

Scalar ScalarPointer::operator*() {
    return deref_mut();
}
Scalar ScalarPointer::operator*() const noexcept {
    return Scalar(*this);
}
ScalarPointer ScalarPointer::operator+(dimn_t index) const noexcept {
    if (p_data == nullptr) {
        return ScalarPointer(p_type);
    }
    assert(p_type != nullptr);
    return {static_cast<const char*>(p_data) + index*p_type->itemsize(), p_type, m_constness};
}
ScalarPointer &ScalarPointer::operator+=(dimn_t index) noexcept {
    if (p_data != nullptr) {
        assert(p_type != nullptr);
        p_data = static_cast<const char*>(p_data) + index*p_type->itemsize();
    }
    return *this;
}
ScalarPointer::difference_type ScalarPointer::operator-(const ScalarPointer &other) const noexcept {
    assert(this->p_type == other.p_type);
    assert(this->p_type != nullptr);
    return static_cast<difference_type>(static_cast<const char*>(p_data) - static_cast<const char*>(other.p_data)) / p_type->itemsize();
}
ScalarPointer &ScalarPointer::operator++() noexcept {
    if (p_data != nullptr) {
        assert(p_type != nullptr);
        p_data = static_cast<const char*>(p_data) + p_type->itemsize();
    }
    return *this;
}
const ScalarPointer ScalarPointer::operator++(int) noexcept {
    ScalarPointer current(*this);
    ++(*this);
    return current;
}

Scalar ScalarPointer::operator[](dimn_t index) {
    if (m_constness == IsConst) {
        throw std::runtime_error("attempting to dereference const pointer to non-const value");
    }
    return Scalar((*this) + index);
}
Scalar ScalarPointer::operator[](dimn_t index) const noexcept {
    return Scalar((*this) + index);
}
