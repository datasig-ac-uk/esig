//
// Created by user on 30/03/2022.
//


#include "esig/intervals.h"

#include <algorithm>
#include <iostream>


namespace esig {


interval::interval() : m_interval_type(interval_type::clopen)
{
}
interval::interval(interval_type itype) : m_interval_type(itype)
{
}
interval_type interval::get_type() const noexcept
{
    return m_interval_type;
}
bool interval::operator==(const interval &rhs) const
{
    return m_interval_type == rhs.m_interval_type && inf() == rhs.inf() && sup() == rhs.sup();
}
bool interval::operator!=(const interval &rhs) const
{
    return !operator==(rhs);
}

std::ostream &operator<<(std::ostream &os, const interval &itvl)
{
    if (itvl.get_type() == interval_type::clopen) {
        os << '[' << itvl.inf() << ", " << itvl.sup() << ')';
    } else {
        os << '(' << itvl.inf() << ", " << itvl.sup() << ']';
    }
    return os;
}

bool interval::contains(param_t arg) const noexcept
{
    if (m_interval_type == interval_type::clopen) {
        return inf() <= arg && arg < sup();
    } else {
        return inf() < arg && arg <= sup();
    }
}


bool interval::is_associated(const interval &arg) const noexcept
{
    return contains(arg.included_end());
}
bool interval::contains(const interval &arg) const noexcept
{
    return is_associated(arg) && !arg.contains(excluded_end());
}
param_t interval::included_end() const
{
    if (m_interval_type == interval_type::clopen) {
        return inf();
    } else {
        return sup();
    }
}
param_t interval::excluded_end() const
{
    if (m_interval_type == interval_type::clopen) {
        return sup();
    } else {
        return inf();
    }
}
bool interval::intersects_with(const interval &arg) const noexcept {
    param_t lhs_inf = inf(), lhs_sup = sup(), rhs_inf = arg.inf(), rhs_sup = arg.sup();

    if ((lhs_inf <= rhs_inf && lhs_sup > rhs_inf) || (rhs_inf <= lhs_inf && rhs_sup > lhs_inf)) {
        // [l--[r---l)--r) || [r--[l--r)--l)
        return true;
    } else if (rhs_inf == lhs_sup) {
        // (l--l][r--r)
        return m_interval_type == interval_type::opencl && arg.m_interval_type == interval_type::clopen;
    } else if (lhs_inf == rhs_sup) {
        // (r--r][l---l)
        return m_interval_type == interval_type::clopen && arg.m_interval_type == interval_type::opencl;
    } else {
        return false;
    }
}

} // namespace esig
