//
// Created by user on 26/04/22.
//

#include <esig/intervals.h>

#include <list>
#include <vector>

namespace esig {


std::vector<dyadic_interval> to_dyadic_intervals(param_t inf, param_t sup, dyadic::power_t tol, interval_type itype)
{
    using iterator = std::list<dyadic_interval>::iterator;
    std::list<dyadic_interval> intervals;

    auto store_move = [&](dyadic_interval &b) {
        intervals.push_back(b.shrink_to_omitted_end());
        b.advance();
    };

    auto store_ = [&](iterator &p, dyadic_interval &e) -> iterator {
        return intervals.insert(p, e.shrink_to_contained_end());
    };

    real_interval real{inf, sup, itype};

    dyadic_interval begin{real.included_end(), tol, itype};
    dyadic_interval end{real.excluded_end(), tol, itype};

    while (!begin.contains(end)) {
        auto next{begin};
        next.expand_interval();
        if (!begin.aligned()) {
            store_move(next);
        }
        begin = std::move(next);
    }

    auto p = intervals.end();
    for (auto next{end}; begin.contains(next.expand_interval());) {
        if (!end.aligned()) {
            p = store_(p, next);
        }
        end = next;
    }

    if (itype == interval_type::opencl) {
        intervals.reverse();
    }

    return {intervals.begin(), intervals.end()};
}

} // namespace esig
