//
// Created by user on 26/04/22.
//

#ifndef ESIG_PATHS_SRC_COMMON_SRC_SEGMENTATION_H_
#define ESIG_PATHS_SRC_COMMON_SRC_SEGMENTATION_H_

#include <esig/implementation_types.h>
#include <esig/intervals.h>
#include <set>
#include <functional>


namespace esig {

using indicator_func = std::function<bool(const interval &)>;


std::vector<real_interval> segment_full(const interval& arg,
                                   indicator_func& in_character,
                                   typename dyadic::power_t trim_tol,
                                   typename dyadic::power_t signal_tol);
std::vector<real_interval> segment_simple(const interval& arg,
                                   indicator_func& in_character,
                                   typename dyadic::power_t precision = 0);


}



#endif//ESIG_PATHS_SRC_COMMON_SRC_SEGMENTATION_H_
