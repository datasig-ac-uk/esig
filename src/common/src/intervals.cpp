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
    return m_interval_type == rhs.m_interval_type;
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


} // namespace esig
