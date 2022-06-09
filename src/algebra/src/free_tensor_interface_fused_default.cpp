//
// Created by user on 17/03/2022.
//


#include <esig/algebra/free_tensor_interface.h>

namespace esig {
namespace algebra {

free_tensor_interface &free_tensor_interface::add_scal_mul(const free_tensor_interface &other, const coefficient &scal)
{
    auto val = other.smul(scal);
    add(*val);
    return *this;
}
free_tensor_interface &free_tensor_interface::sub_scal_mul(const free_tensor_interface &other, const coefficient &scal)
{
    auto val = other.smul(scal);
    sub(*val);
    return *this;
}
free_tensor_interface &free_tensor_interface::add_scal_div(const free_tensor_interface &other, const coefficient &scal)
{
    auto val = other.sdiv(scal);
    add(*val);
    return *this;
}
free_tensor_interface &free_tensor_interface::sub_scal_div(const free_tensor_interface &other, const coefficient &scal)
{
    auto val = other.sdiv(scal);
    sub(*val);
    return *this;
}
free_tensor_interface &free_tensor_interface::add_mul(const free_tensor_interface &lhs, const free_tensor_interface &rhs)
{
    auto val = lhs.mul(rhs);
    add(*val);
    return *this;
}
free_tensor_interface &free_tensor_interface::sub_mul(const free_tensor_interface &lhs, const free_tensor_interface &rhs)
{
    auto val = lhs.mul(rhs);
    sub(*val);
    return *this;
}
free_tensor_interface &free_tensor_interface::mul_smul(const free_tensor_interface &other, const coefficient &scal)
{
    auto val = other.smul(scal);
    mul(*val);
    return *this;
}
free_tensor_interface &free_tensor_interface::mul_sdiv(const free_tensor_interface &other, const coefficient &scal)
{
    auto val = other.sdiv(scal);
    mul(*val);
    return *this;
}

}
}
