//
// Created by user on 26/04/22.
//

#include "segmentation.h"

#include <list>


namespace esig {

namespace {

struct order_by_length
{
    bool operator()(const dyadic_interval& a, const dyadic_interval& b) const noexcept
    {
        return a.power() < b.power();
    }
};





void extend_at_contained_end(std::set<dyadic_interval> &fragmented_interval,
                             typename dyadic::power_t trim_tol,
                             const interval &arg,
                             indicator_func &in_character)
{
    dyadic_interval front = *(fragmented_interval.begin());
    while (front.power() <= trim_tol) {
        real_interval tmp(front.shift_back().included_end(), fragmented_interval.rbegin()->excluded_end());
        while (front.power() <= trim_tol && (!arg.contains(front.shift_back()) || !in_character(tmp))) {
            front = front.shrink_to_contained_end();
        }
        front = front.shift_back();
        if (front.power() > trim_tol) {
            return;
        } else {
            fragmented_interval.insert(front);
        }
    }
}

void extend_at_omitted_end(std::set<dyadic_interval> &fragmented_interval,
                           typename dyadic::power_t trim_tolerance,
                           const interval &arg,
                           indicator_func &in_character)
{
    dyadic_interval back = *fragmented_interval.rbegin();
    back = back.shift_forward();
    while (back.power() <= trim_tolerance) {
        real_interval tmp(fragmented_interval.begin()->included_end(), back.excluded_end());
        while (back.power() <= trim_tolerance && (!arg.contains(back) || !in_character(tmp))) {
            back = back.shrink_to_contained_end();
        }
        if (back.power() > trim_tolerance) {
            return;
        } else {
            fragmented_interval.insert(back);
            back = back.shift_forward();
        }
    }
}

void extend(std::set<dyadic_interval> &fragmented_interval,
            typename dyadic::power_t trim_tolerance,
            const interval &arg,
            indicator_func &in_character)
{
extend_at_contained_end(fragmented_interval, trim_tolerance, arg, in_character);
extend_at_omitted_end(fragmented_interval, trim_tolerance, arg, in_character);
}


std::list<real_interval> segment_impl(const interval& arg, indicator_func& in_character, typename dyadic::power_t trim_tol, typename dyadic::power_t signal_tol)
{
    std::list<real_interval> result;

    std::multiset<dyadic_interval, order_by_length> partition;
    for (auto di : to_dyadic_intervals(arg.inf(), arg.sup(), signal_tol)) {
        partition.insert(di);
    }

    auto p = partition.begin();
    while (p->power() <= signal_tol && !in_character(*p)) {
        dyadic_interval front = *p;
        partition.erase(p);
        partition.emplace(dyadic_interval(front).shrink_to_contained_end());
        partition.emplace(dyadic_interval(front).shrink_to_omitted_end());
        p = partition.begin();
    }

    if (p->power() <= signal_tol) {
        std::set<dyadic_interval> fragmented_interval;
        dyadic_interval central = *p;
        fragmented_interval.insert(central);
        extend(fragmented_interval, trim_tol, arg, in_character);

        result.emplace_back(fragmented_interval.begin()->included_end(),
                            fragmented_interval.rbegin()->excluded_end());

        real_interval argbegin(arg.included_end(), fragmented_interval.begin()->included_end());
        real_interval argend(fragmented_interval.rbegin()->excluded_end(), arg.excluded_end());

        result.splice(result.begin(), segment_impl(argbegin, in_character, trim_tol, signal_tol));
        result.splice(result.end(), segment_impl(argend, in_character, trim_tol, signal_tol));
    }
    return result;
}

} // namespace

std::vector<real_interval>
segment_full(const interval &arg, indicator_func &in_character, dyadic::power_t trim_tol, dyadic::power_t signal_tol)
{
    auto result = segment_impl(arg, in_character, trim_tol, signal_tol);
    return {result.begin(), result.end()};
}
std::vector<real_interval>
segment_simple(const interval &arg, indicator_func &in_character, typename dyadic::power_t precision)
{
    return segment_full(arg, in_character, precision, precision);
}


} // namespace esig
