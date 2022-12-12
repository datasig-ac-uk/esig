//
// Created by user on 02/11/22.
//

#include "esig/scalars.h"


using namespace esig;
using namespace esig::scalars;

scalar_type::~scalar_type() {
}

scalar_type::converter_function scalar_type::get_converter(const std::string& id) const noexcept {
    return nullptr;
}
void scalar_type::register_converter(const std::string& id, scalar_type::converter_function func) const {
}


std::pair<scalar, void *> scalar_type::new_owned_scalar() const {
    scalar s(allocate(1), scalar::OwnedPointer);
    return {s, const_cast<void*>(s.ptr())};
}
const scalar_type* scalar_type::rational_type() const noexcept
{
    return this;
}

scalar scalar_type::one() const {
    return from(1);
}
scalar scalar_type::mone() const {
    return from(-1);
}
scalar scalar_type::zero() const {
    return from(0);
}
bool scalar_type::is_zero(const void * arg) const {
    if (arg == nullptr) {
        return true;
    }
    return are_equal(arg, zero().to_const_pointer());
}
void scalar_type::assign(void *dst, scalar_pointer src, dimn_t count) const {
    auto* ptr = static_cast<char*>(dst);
    auto itemsize = m_info.size;
    for (dimn_t i=0; i<count; ++i) {
        assign(ptr, src);
        ptr += itemsize;
    }
}
void scalar_type::assign(void *dst, long long int numerator, long long int denominator) const
{
    auto tmp = this->from(numerator, denominator);
    assign(dst, tmp.to_const_pointer(), 1);
}
void scalar_type::add_inplace(void *lhs, scalar_pointer rhs) const {
    auto tmp = add(lhs, rhs);
    assign(lhs, tmp.to_const_pointer());
}
void scalar_type::sub_inplace(void * lhs, scalar_pointer rhs) const {
    auto tmp = sub(lhs, rhs);
    assign(lhs, tmp.to_const_pointer());
}
void scalar_type::mul_inplace(void * lhs, scalar_pointer rhs) const {
    auto tmp = mul(lhs, rhs);
    assign(lhs, tmp.to_const_pointer());
}
void scalar_type::div_inplace(void * lhs, scalar_pointer rhs) const {
    auto tmp = div(lhs, rhs);
    assign(lhs, tmp.to_const_pointer());
}

void scalar_type::print(const void * arg, std::ostream &os) const {
    os << to_scalar_t(arg);
}
