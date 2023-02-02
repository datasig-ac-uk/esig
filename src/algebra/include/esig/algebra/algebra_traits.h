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


namespace esig {
namespace algebra {
namespace dtl {

template <typename Basis>
struct basis_info
{
    using this_key_type = typename Basis::key_type;


    static this_key_type convert_key(const Basis& basis, esig::key_type key);
    static esig::key_type convert_key(const Basis& basis, const this_key_type& key);

    static esig::key_type first_key(const Basis& basis);
    static esig::key_type last_key(const Basis& basis);

    static deg_t native_degree(const Basis& basis, const this_key_type& key);
    static deg_t degree(const Basis& basis, esig::key_type key);



};



} // namespace dtl
} // namespace algebra
} // namespace esig


#endif//ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_
