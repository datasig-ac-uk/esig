//
// Created by user on 02/11/22.
//

#include "esig/scalars.h"

using namespace esig;


const scalar_type* esig::scalar_type::rational_type() const noexcept
{
    return this;
}
scalar_type::interface_ptr scalar_type::from(const scalar_interface *other) const {
    if (other == nullptr) {
        return this->zero();
    }
    return this->from(other->as_scalar());
}

scalar_type::interface_ptr scalar_type::one() const {
    return this->from(1);
}
scalar_type::interface_ptr scalar_type::mone() const {
    return this->from(-1);
}
scalar_type::interface_ptr scalar_type::zero() const {
    return this->from(0);
}
