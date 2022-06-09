//
// Created by user on 26/04/22.
//

#include <esig/intervals.h>

#include <cassert>
#include <cmath>
#include <iostream>


namespace esig {

dyadic::multiplier_t dyadic::mod(dyadic::multiplier_t a, dyadic::multiplier_t b)
{
    dyadic::multiplier_t r = a % b;
    return (r < 0) ? (r + abs(b)) : r;
}
dyadic::multiplier_t dyadic::int_two_to_int_power(dyadic::power_t exponent)
{
    assert(exponent >= 0);
    return multiplier_t(1) << exponent;
}
dyadic::multiplier_t dyadic::shift(dyadic::multiplier_t k, dyadic::power_t n)
{
    return k * int_two_to_int_power(n);
}

dyadic::dyadic(dyadic::multiplier_t k) : m_multiplier(k), m_power(0)
{
}
dyadic::dyadic(dyadic::multiplier_t k, dyadic::power_t n)
    : m_multiplier(k), m_power(n)
{
}
dyadic::multiplier_t dyadic::multiplier() const noexcept
{
    return m_multiplier;
}
dyadic::power_t dyadic::power() const noexcept
{
    return m_power;
}
dyadic::operator param_t() const
{
    return ldexp(param_t(m_multiplier), -m_power);
}
dyadic &dyadic::move_forward(dyadic::multiplier_t arg)
{
    m_multiplier += arg;
    return *this;
}
dyadic &dyadic::operator++()
{
    ++m_multiplier;
    return *this;
}
dyadic dyadic::operator++(int)
{
    dyadic old(*this);
    ++m_multiplier;
    return old;
}
dyadic &dyadic::operator--()
{
    --m_multiplier;
    return *this;
}
dyadic dyadic::operator--(int)
{
    dyadic old(*this);
    --m_multiplier;
    return old;
}
bool dyadic::rebase(dyadic::power_t resolution)
{
    if (m_multiplier == 0) {
        m_power = resolution;
        return true;
    } else if (resolution >= m_power) {
        m_multiplier = dyadic::shift(m_multiplier, resolution - m_power);
        m_power = resolution;
        return true;
    }

    if (m_power >= std::numeric_limits<dyadic::multiplier_t>::digits + resolution) {
        resolution = (m_power - std::numeric_limits<dyadic::power_t>::digits) + 1;
    }

    power_t rel_resolution{m_power - resolution};


    // starting at relative resolution find the first n in decreasing order so that 2^n divides k
    // 2^0 always divides k so the action stops
    power_t r = rel_resolution;
    for (; (m_multiplier % dyadic::int_two_to_int_power(r)) != 0; --r) {}

    power_t offset = r;
    m_multiplier /= int_two_to_int_power(offset);
    m_power -= offset;
    //pr();
    return resolution == m_power;
}
bool dyadic::operator<(const dyadic &rhs) const
{
    return (m_power <= rhs.m_power) ? (m_multiplier < dyadic::shift(rhs.m_multiplier, rhs.m_power - m_power)) : dyadic::shift(m_multiplier, (m_power - rhs.m_power)) < rhs.m_multiplier;
}
bool dyadic::operator>(const dyadic &rhs) const
{
    return (m_power <= rhs.m_power) ? (m_multiplier > dyadic::shift(rhs.m_multiplier, rhs.m_power - m_power)) : dyadic::shift(m_multiplier, (m_power - rhs.m_power)) > rhs.m_multiplier;
}
bool dyadic::operator<=(const dyadic &rhs) const
{
    return (m_power <= rhs.m_power) ? (m_multiplier <= dyadic::shift(rhs.m_multiplier, rhs.m_power - m_power)) : dyadic::shift(m_multiplier, (m_power - rhs.m_power)) <= rhs.m_multiplier;
}
bool dyadic::operator>=(const dyadic &rhs) const
{
    return (m_power <= rhs.m_power) ? (m_multiplier >= dyadic::shift(rhs.m_multiplier, rhs.m_power - m_power)) : dyadic::shift(m_multiplier, (m_power - rhs.m_power)) >= rhs.m_multiplier;
}
std::ostream &operator<<(std::ostream &os, const dyadic &di)
{
    return os << '(' << di.multiplier() << ", " << di.power() << ')';
}
bool dyadic_equals(const dyadic &lhs, const dyadic &rhs)
{
    return lhs.multiplier() == rhs.multiplier() && lhs.power() == rhs.power();
}
bool rational_equals(const dyadic &lhs, const dyadic &rhs)
{
    dyadic::multiplier_t ratio;
    if (lhs.multiplier() % rhs.multiplier() == 0 &&
        (ratio = lhs.multiplier() / rhs.multiplier()) >= 1) {
        dyadic::power_t rel_tolerance = lhs.power() - rhs.power();
        if (rel_tolerance < 0) {
            return false;
        }
        return ratio == dyadic::int_two_to_int_power(rel_tolerance);
    } else if (rhs.multiplier() % lhs.multiplier() == 0 &&
               (ratio = rhs.multiplier() / lhs.multiplier()) >= 1) {
        dyadic::power_t rel_tolerance = rhs.power() - lhs.power();
        if (rel_tolerance < 0) {
            return false;
        }
        return ratio == dyadic::int_two_to_int_power(rel_tolerance);
    }
    return false;
}

dyadic_interval dyadic_interval::shift_forward(dyadic::power_t arg) const
{
    dyadic_interval tmp(*this);
    tmp.m_multiplier -= unit() * arg;
    return tmp;
}
dyadic_interval dyadic_interval::shift_back(dyadic::power_t arg) const
{
    dyadic_interval tmp(*this);
    tmp.m_multiplier += unit() * arg;
    return tmp;
}


} // namespace esig
