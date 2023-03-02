//
// Created by user on 22/07/22.
//

#include "esig/paths/piecewise_lie_path.h"

#include <algorithm>
#include <cmath>

namespace esig {
namespace paths {

algebra::Lie piecewise_lie_path::compute_lie_piece(const piecewise_lie_path::lie_piece &arg, const interval &domain)
{
    auto sf = (domain.sup() - domain.inf()) / (arg.first.sup() - arg.first.inf());
    return arg.second.smul(scalars::Scalar(sf));
}

piecewise_lie_path::piecewise_lie_path(std::vector<lie_piece> data, path_metadata metadata)
    : path_interface(std::move(metadata)), m_data()
{
    // first sort so we know the inf of each interval are in order
    auto sort_fun = [](const lie_piece& a, const lie_piece& b) {
        return a.first.inf() < b.first.inf();
    };
    std::sort(data.begin(), data.end(), sort_fun);

    m_data.reserve(data.size());
    auto next = data.begin();
    auto it = next++;
    auto end = data.end();
    while (next != data.end()) {
        auto curr_i = it->first.inf(), curr_s = it->first.sup();
        auto next_i = next->first.inf(), next_s = next->first.sup();
        if (next_i < curr_s) {
            /*
             * If the interval of the next piece overlaps with the current piece interval
             * then we need to do a little hacking/slashing to make sure our data is correct.
             */
            real_interval new_curr_interval(curr_i, next_i);
            auto new_lie = compute_lie_piece(*it, new_curr_interval);
            it->first = new_curr_interval;
            it->second.sub_inplace(new_lie);
            m_data.emplace_back(std::move(new_curr_interval), std::move(new_lie));
            /*
             * At this stage the part [a---[b has been sorted , so we need to decide what the next
             * iteration will look like. This means separating the intersecting parts of [b---a)
             * and on a)---b) (if it exists).
             */

            if (curr_s < next_s) {
                // [a---[b--a)----b)
                // it->second is going to be replaced by the part on [b--a) and
                // next->second replace by the part on a)----b)
                real_interval next_curr_interval(next_i, curr_s);
                new_lie = compute_lie_piece(*next, next_curr_interval);

                it->first = real_interval(next_i, curr_s);
                it->second.add_inplace(new_lie);
                next->first = real_interval(curr_s, next_s);
                next->second.sub_inplace(new_lie);

                // Increment next so the next iteration checks the interaction of the new it with next++
                // Do not increment it yet.
                ++next;
                continue;
            } else if (curr_s == next_s) {
                // [a---[b-------ab)
                // it->second is to be discarded, next->second replaced by part on [b----ab).
                next->second.add_inplace(it->second);
            } else {
                // [a---[b---b)---a)
                // it->second replaced by part on [b---b)
                // next-second replaced by part on b)---a)
                new_lie = compute_lie_piece(*it, next->first);
                std::swap(*it, *next);

                it->second.add_inplace(new_lie);
                next->first = real_interval(next_s, curr_s);
                next->second.sub_inplace(new_lie);

                // Increment next so the next iteration checks the interaction of the new it with next++
                // Do not increment it yet.
                ++next;
                continue;
            }
        } else {
            // Here the intervals of definition are disjoint, so we can just push onto m_data.
            m_data.push_back(std::move(*it));
        }

        if (++it == next) {
            ++next;
        }
    }
    for (; it != end; ++it) {
        m_data.push_back(std::move(*it));
    }



}
bool piecewise_lie_path::empty(const interval &domain) const {
    return path_interface::empty(domain);
}
algebra::Lie piecewise_lie_path::log_signature(const interval &domain, const algebra::context &ctx) const {
    std::vector<algebra::Lie> lies;
    lies.reserve(4);

    auto a = domain.inf(), b = domain.sup();
    for (const auto& piece : m_data) {
        // data is in order, so if we are already past the end of the request interval,
        // then we are done so break.
        auto pa = piece.first.inf(), pb = piece.first.sup();
        if (pa >= b) {
            // [-----) [p----p)
            break;
        }
        if (pb <= a) {
            // [p----p) [-----)
        } else if (pa >= a && pb <= b) {
            // [-----[p-----p)---)
            lies.push_back(piece.second);
        } else if (pb <= pb) {
            // [p---[---p)-----)
            real_interval sub_domain (a, pb);
            lies.push_back(compute_lie_piece(piece, sub_domain));
        } else if (pa >= a && pb > b) {
            // [---[p----)----p)
            real_interval sub_domain (pa, b);
            lies.push_back(compute_lie_piece(piece, sub_domain));
        }
    }

    const auto& md = metadata();
    return ctx.cbh(lies, md.result_vec_type);
}
}// namespace paths
}// namespace esig
