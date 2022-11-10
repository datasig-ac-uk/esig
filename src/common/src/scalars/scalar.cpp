//
// Created by user on 03/11/22.
//

#include "esig/scalars.h"


using namespace esig;


scalar::scalar(const scalar_type *type) : p_impl(type->zero()){}
scalar::scalar(scalar_t) {}
scalar::scalar(scalar_t, const scalar_type *) {}

scalar::scalar(const scalar &other) : p_impl(other.type()->from(other.ptr()))
{}
scalar::scalar(scalar &&other) noexcept : p_impl(std::move(other.p_impl))
{}

scalar &scalar::operator=(const scalar &other) {
    if (&other != this) {
        if (p_impl) {
            p_impl->assign(other.ptr());
        } else if (other.p_impl) {
            p_impl = other.ptr()->type()->from(other.ptr());
        } else {
            p_impl = nullptr;
        }
    }
    return *this;
}
scalar &scalar::operator=(scalar &&other) noexcept {
    if (&other != this) {
        p_impl = std::move(other.p_impl);
    }
    return *this;
}

scalar scalar::operator-() const {
    if (p_impl) {
        return {p_impl->uminus()};
    }
    return scalar(nullptr);
}
scalar scalar::operator+(const scalar &other) const {
    return {p_impl->add(other.ptr())};
}
scalar scalar::operator-(const scalar &other) const {
    return {p_impl->sub(other.ptr())};
}
scalar scalar::operator*(const scalar &other) const {
    return {p_impl->mul(other.ptr())};
}
scalar scalar::operator/(const scalar &other) const {
    return {p_impl->div(other.ptr())};
}
scalar& scalar::operator+=(const scalar &other) {
    if (is_zero()) {
        operator=(other);
    } else if (!other.is_zero()) {
        p_impl->add_inplace(other.ptr());
    }
    return *this;
}
scalar& scalar::operator-=(const scalar &other) {
    if (is_zero()) {
        operator=(-other);
    } else if (!other.is_zero()) {
        p_impl->sub_inplace(other.ptr());
    }
    return *this;
}
scalar& scalar::operator*=(const scalar &other) {
    if (is_zero() || other.is_zero()) {
        
    }
    p_impl->mul_inplace(other.ptr());
    return *this;
}
scalar& scalar::operator/=(const scalar &other) {
    p_impl->div_inplace(other.ptr());
    return *this;
}
bool scalar::operator==(const scalar &rhs) const noexcept {
    return false;
}
bool scalar::operator!=(const scalar &rhs) const noexcept {
    return false;
}
