//
// Created by user on 03/11/22.
//

#include "esig/scalars.h"

using namespace esig;

scalar scalar_interface::uminus() const {
    auto tmp = this->type()->mone();
    return this->mul(tmp.get());
}
void scalar_interface::add_inplace(const scalar_interface *other)
{
    this->assign(this->add(other).p_impl.get());
}
void scalar_interface::sub_inplace(const scalar_interface *other) {
    this->assign(this->sub(other).p_impl.get());
}
void scalar_interface::mul_inplace(const scalar_interface *other) {
    this->assign(this->mul(other).p_impl.get());
}
void scalar_interface::div_inplace(const scalar_interface *other) {
    this->assign(this->div(other).p_impl.get());
}

bool scalar_interface::equals(const scalar_interface *other) const noexcept {
    return this->as_scalar() == other->as_scalar();

}
std::ostream &scalar_interface::print(std::ostream &os) const {
    os << this->as_scalar();
    return os;
}
