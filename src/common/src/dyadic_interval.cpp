//
// Created by user on 26/04/22.
//
#include <esig/intervals.h>

#include <cassert>
#include <cmath>
#include <iostream>

namespace esig {


dyadic_interval::dyadic_interval() : dyadic{0, 0}, interval(interval_type::clopen)
{
}
dyadic_interval::dyadic_interval(dyadic::multiplier_t k) : dyadic(k)
{
}
dyadic_interval::dyadic_interval(dyadic::multiplier_t k, dyadic::power_t n) : dyadic(k, n)
{
}
dyadic_interval::dyadic_interval(dyadic::multiplier_t k, dyadic::power_t n, interval_type itype)
    : dyadic{k, n}, interval(itype)
{
}
dyadic_interval::dyadic_interval(interval_type itype) : dyadic{0, 0}, interval(itype)
{
}
dyadic_interval::dyadic_interval(dyadic di) : dyadic(di)
{
}
dyadic_interval::dyadic_interval(dyadic di, interval_type itype) : dyadic(di), interval(itype)
{
}
dyadic_interval::dyadic_interval(dyadic di, dyadic::power_t resolution, interval_type itype)
    : interval(itype), dyadic(di)
{
    if (!rebase(resolution)) {
        multiplier_t k1 = m_multiplier;
        const multiplier_t one = unit();
        multiplier_t pow = dyadic::int_two_to_int_power(m_power - resolution);
        m_multiplier = one * (k1 * one - mod(k1 * one, pow));
        bool is_int = rebase(resolution);
        assert(is_int);
    }
}
dyadic_interval::dyadic_interval(param_t val, dyadic::power_t resolution, interval_type itype)
    : dyadic{0, 0}, interval(itype)
{
    auto rescaled = ldexp(val, resolution);
    if (m_interval_type == interval_type::opencl) {
        m_multiplier = ceil(rescaled);
    } else {
        m_multiplier = floor(rescaled);
    }
    m_power = resolution;
}

dyadic::multiplier_t dyadic_interval::unit() const noexcept
{
    if (m_interval_type == interval_type::clopen) {
        return 1;
    } else {
        return -1;
    }
}
param_t dyadic_interval::included_end() const
{
    return static_cast<param_t>(dincluded_end());
}
param_t dyadic_interval::excluded_end() const
{
    return static_cast<param_t>(dexcluded_end());
}
param_t dyadic_interval::inf() const
{
    return static_cast<param_t>(dinf());
}
param_t dyadic_interval::sup() const
{
    return static_cast<param_t>(dsup());
}
dyadic dyadic_interval::dincluded_end() const
{
    return static_cast<const dyadic &>(*this);
}
dyadic dyadic_interval::dexcluded_end() const
{
    return {m_multiplier + unit(), m_power};
}
dyadic dyadic_interval::dinf() const
{
    if (m_interval_type == interval_type::clopen) {
        return dincluded_end();
    } else {
        return dexcluded_end();
    }
}
dyadic dyadic_interval::dsup() const
{
    if (m_interval_type == interval_type::clopen) {
        return dexcluded_end();
    } else {
        return dincluded_end();
    }
}
dyadic_interval dyadic_interval::shrink_to_contained_end(dyadic::power_t arg) const
{
    return {static_cast<const dyadic &>(*this), arg + m_power, m_interval_type};
}
dyadic_interval dyadic_interval::shrink_to_omitted_end() const
{
    return shrink_to_contained_end().flip_interval();
}
dyadic_interval &dyadic_interval::shrink_interval_right()
{
    if (m_interval_type == interval_type::opencl) {
        *this = shrink_to_contained_end();
    } else {
        *this = shrink_to_omitted_end();
    }
    return *this;
}
//dyadic_interval &dyadic_interval::shrink_interval_left()
//{
//    if (m_interval_type == interval_type::clopen) {
//        *this = shrink_to_contained_end();
//    } else {
//        *this = shrink_to_omitted_end();
//    }
//    return *this;
//}
dyadic_interval &dyadic_interval::shrink_interval_left(dyadic::power_t k)
{
    assert(k >= 0);
    for (; k > 0; --k) {
            if (m_interval_type == interval_type::clopen) {
                *this = shrink_to_contained_end();
            } else {
                *this = shrink_to_omitted_end();
            }
    }
    return *this;
}
dyadic_interval &dyadic_interval::expand_interval(dyadic::multiplier_t arg)
{
    *this = dyadic_interval{dincluded_end(), m_power - arg};
    return *this;
}
bool dyadic_interval::contains(const dyadic_interval &other) const
{
    if (other.m_interval_type != m_interval_type) {
        return false;
    }
    if (other.m_power >= m_power) {
        multiplier_t one(unit());
        multiplier_t pow = int_two_to_int_power(other.m_power - m_power);
        multiplier_t shifted = shift(m_multiplier, (other.m_power - m_power));
        multiplier_t aligned = one * (other.m_multiplier * one - mod(other.m_multiplier * one, pow));

        return shifted == aligned;
    } else {
        return false;
    }
}
bool dyadic_interval::aligned() const
{
    dyadic_interval parent{static_cast<const dyadic &>(*this), m_power - 1, m_interval_type};
    return operator==(dyadic_interval{static_cast<const dyadic &>(parent), m_power});
}
dyadic_interval &dyadic_interval::flip_interval()
{
    if ((m_multiplier % 2) == 0) {
        m_multiplier += unit();
    } else {
        m_multiplier -= unit();
    }
    return *this;
}
bool dyadic_interval::operator==(const interval &other) const
{
    try {
        auto &dother = dynamic_cast<const dyadic_interval &>(other);
        return dyadic_equals(*this, dother) && interval::operator==(other);
    } catch (std::bad_cast &) {
        return false;
    }
}
std::ostream &operator<<(std::ostream &os, const dyadic_interval &di)
{
    return os << static_cast<const interval &>(di);
}
dyadic_interval &dyadic_interval::advance() noexcept
{
    m_multiplier += unit();
    return *this;
}

} // namespace esig
