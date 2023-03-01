//
// Created by user on 02/11/22.
//

#include "esig/scalar_type.h"
#include "esig/scalar.h"
#include "esig/scalar_traits.h"

using namespace esig;
using namespace esig::scalars;

ScalarType::~ScalarType() {
}


const ScalarType *ScalarType::rational_type() const noexcept
{

    return this;
}

Scalar ScalarType::one() const {
    return from(1);
}
Scalar ScalarType::mone() const {
    return from(-1);
}
Scalar ScalarType::zero() const {
    return from(0);
}
bool ScalarType::is_zero(const void * arg) const {
    if (arg == nullptr) {
        return true;
    }
    return are_equal(arg, zero().to_const_pointer());
}
void ScalarType::assign(void *dst, ScalarPointer src, dimn_t count) const {
    auto* ptr = static_cast<char*>(dst);
    auto itemsize = m_info.size;
    for (dimn_t i=0; i<count; ++i) {
        assign(ptr, src);
        ptr += itemsize;
    }
}
void ScalarType::assign(void *dst, long long int numerator, long long int denominator) const
{
    auto tmp = this->from(numerator, denominator);
    assign(dst, tmp.to_const_pointer(), 1);
}
void ScalarType::add_inplace(void *lhs, ScalarPointer rhs) const {
    auto tmp = add(lhs, rhs);
    assign(lhs, tmp.to_const_pointer());
}
void ScalarType::sub_inplace(void * lhs, ScalarPointer rhs) const {
    auto tmp = sub(lhs, rhs);
    assign(lhs, tmp.to_const_pointer());
}
void ScalarType::mul_inplace(void * lhs, ScalarPointer rhs) const {
    auto tmp = mul(lhs, rhs);
    assign(lhs, tmp.to_const_pointer());
}
void ScalarType::div_inplace(void * lhs, ScalarPointer rhs) const {
    auto tmp = div(lhs, rhs);
    assign(lhs, tmp.to_const_pointer());
}

void ScalarType::print(const void * arg, std::ostream &os) const {
    os << to_scalar_t(arg);
}

const ScalarType *ScalarType::from_type_details(const BasicScalarTypeDetails &details) {

    switch (details.code) {
        case 0U: // signed int
        case 1U: // unsigned int
            return ScalarType::of<double>();
        case 2U: // float
            switch (details.bits) {
                case 32:
                    return ScalarType::of<float>();
                case 64:
                    return ScalarType::of<double>();
            }
        case 3U: // Opaque handle
        case 4U: // bfloat
        case 5U: // Complex
            throw std::invalid_argument("no matching type");
        case 6U: // Bool
            return ScalarType::of<double>();
    }

    throw std::invalid_argument("no matching type");
}
const ScalarType *ScalarType::for_id(const std::string &id) {

    try {
        return get_type(id);
    } catch (...) {
        return ScalarType::of<double>();
    }

}
void ScalarType::convert_copy(ScalarPointer out, const void *in, dimn_t count, const BasicScalarTypeDetails &details) const {
}
