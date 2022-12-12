//
// Created by user on 19/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_
#define ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_

#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>

#include <cassert>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

namespace esig {
namespace algebra {



template <typename Algebra>
struct dense_data_access
{
    // Should be defined in specialisations.
    std::pair<const void*, const void*> starting_at(const Algebra& alg, key_type k)
    {
        return {nullptr, nullptr};
    }
};



namespace dtl {


} // namespace dtl



} // namespace algebra
} // namespace esig


#endif//ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_
