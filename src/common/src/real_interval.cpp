//
// Created by user on 26/04/22.
//

#include <esig/intervals.h>

#include <cassert>
#include <cmath>
#include <list>

using namespace esig;


real_interval::real_interval() : m_inf(0.0), m_sup(1.0)
{
}
real_interval::real_interval(param_t inf, param_t sup) : m_inf(inf), m_sup(sup)
{
    assert(inf <= sup);
}
real_interval::real_interval(param_t inf, param_t sup, interval_type itype)
        : interval(itype), m_inf(inf), m_sup(sup)
{
}
real_interval::real_interval(const interval &itvl) : m_inf(itvl.inf()), m_sup(itvl.sup())
{
}
param_t real_interval::included_end() const
{
    if (m_interval_type == interval_type::clopen) {
        return m_inf;
    } else {
        return m_sup;
    }
}
param_t real_interval::excluded_end() const
{
    if (m_interval_type == interval_type::clopen) {
        return m_sup;
    } else {
        return m_inf;
    }
}
param_t real_interval::inf() const
{
    return m_inf;
}
param_t real_interval::sup() const
{
    return m_sup;
}

bool real_interval::contains(param_t arg) const noexcept
{
    if (m_interval_type == interval_type::clopen) {
        return m_inf <= arg && arg < m_sup;
    } else {
        return m_inf < arg && arg <= m_sup;
    }
}
real_interval::real_interval(interval_type itype) : interval(itype), m_inf(0), m_sup(1)
{
}
