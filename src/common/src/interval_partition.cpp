//
// Created by user on 31/10/22.
//

#include "esig/intervals.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace esig {

partition::partition(real_interval base, std::initializer_list<param_t> midpoints)
    : real_interval(std::move(base))
{
    const auto inf = real_interval::inf();
    const auto sup = real_interval::sup();
    m_midpoints.reserve(midpoints.size());
    for (auto point : midpoints) {
        if (inf < point && point < sup) {
            m_midpoints.push_back(point);
        }
    }
    std::sort(m_midpoints.begin(), m_midpoints.end());
}

real_interval partition::operator[](dimn_t index) const noexcept {
    assert(index < m_midpoints.size());
    if (m_midpoints.empty()) {
        return {*this};
    }
    if (index == 0) {
        return {real_interval::inf(), m_midpoints[1]};
    }
    if (index == m_midpoints.size() - 1) {
        return {m_midpoints.back(), real_interval::sup()};
    }

    return {m_midpoints[index], m_midpoints[index + 1]};
}

}// namespace esig
