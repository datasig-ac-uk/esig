//
// Created by user on 24/03/2022.
//

#include <esig/algebra/coefficients.h>

namespace esig {
namespace algebra {

scalar_t coefficient_interface::as_scalar() const
{
    throw std::runtime_error("cannot convert to scalar");
}
void coefficient_interface::assign(coefficient val)
{
    throw std::runtime_error("cannot assign to this value");
}
coefficient coefficient_interface::add(const coefficient_interface &other) const
{
    throw std::runtime_error("cannot add this value");
}
coefficient coefficient_interface::sub(const coefficient_interface &other) const
{
    throw std::runtime_error("cannot subtract this value");
}
coefficient coefficient_interface::mul(const coefficient_interface &other) const
{
    throw std::runtime_error("cannot multiply this value");
}
coefficient coefficient_interface::div(const coefficient_interface &other) const
{
    throw std::runtime_error("cannot divide this value");
}
bool coefficient_interface::equals(const coefficient_interface &other) const noexcept {
    return false;
}
coefficient coefficient::operator-() const
{
    throw std::runtime_error("cannot unary minus this value");
}

coefficient::coefficient() : p_impl(nullptr)
{
}

coefficient::coefficient(coefficient_type ctype) : p_impl(nullptr)
{
}

coefficient::coefficient(param_t arg) : p_impl(new dtl::coefficient_implementation<double>(arg))
{
}
coefficient::coefficient(param_t arg, coefficient_type ctype)
{
}
coefficient::coefficient(long n, long d, coefficient_type ctype)
{
}
coefficient::coefficient(long long int n, long long int d, coefficient_type ctype) {
}
coefficient::coefficient(std::shared_ptr<coefficient_interface> &&arg)
    : p_impl(std::move(arg))
{
}

coefficient::operator scalar_t() const noexcept
{
    return p_impl->as_scalar();
}
const coefficient_interface &coefficient::operator*() const noexcept
{
    return *p_impl;
}
std::ostream &coefficient_interface::print(std::ostream &os) const
{
    return os;
}

bool coefficient::is_const() const noexcept
{
    return false;
}
bool coefficient::is_value() const noexcept
{
    return false;
}
bool coefficient::is_zero() const noexcept {
    if (p_impl) {
        return p_impl->is_zero();
    }
    return true;
}
coefficient_type coefficient::ctype() const noexcept
{
    return coefficient_type::sp_real;
}

coefficient &coefficient::operator=(const coefficient &other)
{
    p_impl->assign(other);
    return *this;
}
coefficient &coefficient::operator+=(const coefficient &other)
{
    p_impl->assign(p_impl->add(*other.p_impl));
    return *this;
}
coefficient &coefficient::operator-=(const coefficient &other)
{
    p_impl->assign(p_impl->sub(*other.p_impl));
    return *this;
}
coefficient &coefficient::operator*=(const coefficient &other)
{
    p_impl->assign(p_impl->mul(*other.p_impl));
    return *this;
}
coefficient &coefficient::operator/=(const coefficient &other)
{
    p_impl->assign(p_impl->div(*other.p_impl));
    return *this;
}

coefficient coefficient::operator+(const coefficient &other) const
{
    return coefficient(p_impl->add(*other.p_impl));
}
coefficient coefficient::operator-(const coefficient &other) const
{
    return coefficient(p_impl->sub(*other.p_impl));
}
coefficient coefficient::operator*(const coefficient &other) const
{
    return coefficient(p_impl->mul(*other.p_impl));
}
coefficient coefficient::operator/(const coefficient &other) const
{
    return coefficient(p_impl->div(*other.p_impl));
}


coefficient &coefficient::operator+=(const scalar_t &other)
{
    p_impl->assign(p_impl->add(other));
    return *this;
}
coefficient &coefficient::operator-=(const scalar_t &other)
{
    p_impl->assign(p_impl->sub(other));
    return *this;
}
coefficient &coefficient::operator*=(const scalar_t &other)
{
    p_impl->assign(p_impl->mul(other));
    return *this;
}
coefficient &coefficient::operator/=(const scalar_t &other)
{
    p_impl->assign(p_impl->div(other));
    return *this;
}

bool coefficient::operator==(const coefficient &rhs) const noexcept {
    if (!p_impl) {
        return rhs.is_zero();
    }
    if (!rhs.p_impl) {
        return is_zero();
    }
    return p_impl->equals(*rhs.p_impl);
}
#define ESIG_ALGEBRA_IMPLEMENT_METHOD(NAME)                                    \
    coefficient coefficient_interface::NAME(const scalar_t &other) const \
    {                                                                          \
        switch (ctype()) {                                                     \
            case coefficient_type::sp_real: {                                  \
                auto cast = static_cast<float>(other);                         \
                dtl::coefficient_implementation<float> wrapper(cast);          \
                return NAME(wrapper);                                          \
            }                                                                  \
            case coefficient_type::dp_real: {                                  \
                auto cast = static_cast<double>(other);                        \
                dtl::coefficient_implementation<double> wrapper(cast);         \
                return NAME(wrapper);                                          \
            }                                                                  \
        }                                                                      \
        throw std::bad_cast();                                                 \
    }

ESIG_ALGEBRA_IMPLEMENT_METHOD(add)
ESIG_ALGEBRA_IMPLEMENT_METHOD(sub)
ESIG_ALGEBRA_IMPLEMENT_METHOD(mul)
ESIG_ALGEBRA_IMPLEMENT_METHOD(div)

#undef ESIG_ALGEBRA_IMPLEMENT_METHOD

}// namespace algebra
}// namespace esig
