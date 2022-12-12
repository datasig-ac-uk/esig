//
// Created by user on 03/11/22.
//

#include "esig/scalars.h"
#include <ostream>

using namespace esig;
using namespace esig::scalars;

scalar scalar_interface::uminus() const {
    return type()->uminus(to_pointer());
}
//scalar scalar_interface::add(const scalar& other) const {
//    return type()->add(to_pointer().ptr(), other.to_const_pointer());
//}
//scalar scalar_interface::sub(const scalar& other) const {
//    return type()->sub(to_pointer().ptr(), other.to_const_pointer());
//}
//scalar scalar_interface::mul(const scalar& other) const {
//    return type()->mul(to_pointer().ptr(), other.to_const_pointer());
//}
//scalar scalar_interface::div(const scalar& other) const {
//    return type()->div(to_pointer().ptr(), other.to_const_pointer());
//}
bool scalar_interface::equals(const scalar& other) const noexcept {
    return type()->are_equal(to_pointer().ptr(), other.to_const_pointer());
}


std::ostream &scalar_interface::print(std::ostream &os) const {
    os << this->as_scalar();
    return os;
}
