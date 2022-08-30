//
// Created by user on 19/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_
#define ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_

#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/coefficients.h>

#include <cassert>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

namespace esig {
namespace algebra {


enum class vector_type
{
    sparse,
    dense
};



namespace dtl {
template<typename Iterator>
struct iterator_traits {
    static key_type key(Iterator it) noexcept
    { return it->first; }
    static typename std::iterator_traits<Iterator>::value_type::second_type
    value(Iterator it) noexcept
    { return it->second; }
};


template<typename It>
struct multiplication_data {
    It begin;
    It end;
    std::vector<It> degree_ranges;
    deg_t degree;
};


template<typename Algebra, typename LhsIt, typename RhsIt, typename Op>
void fallback_multiplication(Algebra &result,
                             const multiplication_data<LhsIt>& lhs,
                             const multiplication_data<RhsIt>& rhs,
                             Op op);

template <typename Algebra, typename LhsIt, typename RhsIt>
struct multiplication_dispatcher
{
    template <typename Op>
    static void dispatch(Algebra& result,
                         const multiplication_data<LhsIt>& lhs,
                         const multiplication_data<RhsIt>& rhs,
                         Op op)
    {
        dtl::fallback_multiplication(result, lhs, rhs, op);
    }
};


} // namespace dtl

template <typename Algebra>
struct algebra_info
{
    using scalar_type = typename Algebra::scalar_type;
    using rational_type = scalar_type;

    static constexpr coefficient_type ctype() noexcept;
    static constexpr vector_type vtype() noexcept;
    static deg_t width(const Algebra* instance) noexcept;
    static deg_t max_depth(const Algebra* instance) noexcept;

    using this_key_type = key_type;
    static this_key_type convert_key(const Algebra* instance, esig::key_type key) noexcept;

    static deg_t degree(const Algebra* instance, esig::key_type key) noexcept;
    static deg_t native_degree(const Algebra* instance, this_key_type key);

    static key_type first_key(const Algebra* instance) noexcept;
    static key_type last_key(const Algebra* instance) noexcept;

    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;

    static const key_prod_container& key_product(const Algebra* inst, key_type k1, key_type k2);

    static Algebra create_like(const Algebra& instance);
};


template<typename Algebra>
constexpr coefficient_type algebra_info<Algebra>::ctype() noexcept
{
    return get_coeff_type(scalar_type(0));
}
template<typename Algebra>
constexpr vector_type algebra_info<Algebra>::vtype() noexcept
{
    return Algebra::vtype;
}
template<typename Algebra>
deg_t algebra_info<Algebra>::width(const Algebra* instance) noexcept
{
    return instance->width();
}
template<typename Algebra>
deg_t algebra_info<Algebra>::max_depth(const Algebra* instance) noexcept
{
    return instance->depth();
}

template<typename Algebra>
deg_t algebra_info<Algebra>::degree(const Algebra *instance, esig::key_type key) noexcept {
    return 0;
}
template<typename Algebra>
deg_t algebra_info<Algebra>::native_degree(const Algebra *instance, algebra_info::this_key_type key) {
    return 0;
}
template<typename Algebra>
key_type algebra_info<Algebra>::convert_key(const Algebra* inst, esig::key_type key) noexcept
{
    return key;
}

template <typename Algebra>
key_type algebra_info<Algebra>::first_key(const Algebra *instance) noexcept
{
    return 0;
}
template<typename Algebra>
key_type algebra_info<Algebra>::last_key(const Algebra *instance) noexcept {
    return instance->basis()->size();
}
template<typename Algebra>
Algebra algebra_info<Algebra>::create_like(const Algebra &instance) {
    return Algebra();
}

template<typename Algebra>
const typename algebra_info<Algebra>::key_prod_container &algebra_info<Algebra>::key_product(const Algebra *inst, key_type k1, key_type k2) {
    static boost::container::small_vector<std::pair<key_type, int>, 0> null;
    return null;
}


struct algebra_base_access
{
    template <typename Wrapper>
    static const typename Wrapper::interface_t* get(const Wrapper& wrap)
    {
        return wrap.p_impl.get();
    }

    template<typename Wrapper>
    static typename Wrapper::interface_t* get(Wrapper &wrap)
    {
        return wrap.p_impl.get();
    }
};



struct algebra_implementation_access
{
    template<template <typename> class Wrapper, typename Impl>
    using interface_t = typename Wrapper<Impl>::interface_t;

    template <template <typename> class Wrapper, typename Impl>
    using algebra_t = typename Wrapper<Impl>::algebra_t;

    template <template <typename> class Wrapper, typename Impl>
    static const Impl& get(const interface_t<Wrapper, Impl>& arg)
    {
        return dynamic_cast<const Wrapper<Impl>&>(arg).m_data;
    }

    template <template <typename> class Wrapper, typename Impl>
    static Impl& get(interface_t<Wrapper, Impl>& arg)
    {
        return dynamic_cast<Wrapper<Impl>&>(arg).m_data;
    }

    template <template <typename> class Wrapper, typename Impl>
    static const Impl& get(const Wrapper<Impl>& arg)
    {
        return arg.m_data;
    }

    template <template <typename> class Wrapper, typename Impl>
    static Impl& get(Wrapper<Impl>& arg)
    {
        return arg.m_data;
    }

    template <template <typename> class Wrapper, typename Impl>
    static const Impl& get(const algebra_t<Wrapper, Impl>& arg)
    {
        return get<Wrapper, Impl>(*arg.p_impl);
    }

    template <template <typename> class Wrapper, typename Impl>
    static Impl& get(algebra_t<Wrapper, Impl>& arg)
    {
        return get<Wrapper, Impl>(*arg.p_impl);
    }

};






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
template<typename Algebra, typename LhsIt, typename RhsIt, typename Op>
void fallback_multiplication(Algebra &result,
                             const multiplication_data<LhsIt>& lhs,
                             const multiplication_data<RhsIt>& rhs,
                             Op op)
{
    using litraits = iterator_traits<LhsIt>;
    using ritraits = iterator_traits<RhsIt>;
    using info = algebra_info<Algebra>;

    auto max_deg = std::min(info::max_depth(&result), lhs.degree + rhs.degree);

    const auto* ref = &result;
    auto product = [ref](LhsIt lit, RhsIt rit)
        -> const typename info::key_prod_container& {
        auto lkey = litraits::key(lit);
        auto rkey = ritraits::key(rit);
        return info::key_product(ref, lkey, rkey);
    };


    for (int out_deg = static_cast<int>(max_deg); out_deg > 0; --out_deg) {
        int lhs_max_deg = std::min(out_deg, static_cast<int>(lhs.degree));
        int lhs_min_deg = std::max(0, out_deg - static_cast<int>(rhs.degree));

        for (int lhs_d = lhs_max_deg; lhs_d >= lhs_min_deg; --lhs_d) {
            auto rhs_d = out_deg - lhs_d;

            for (auto lit = lhs.degree_ranges[lhs_d]; lit != lhs.degree_ranges[lhs_d + 1]; ++lit) {
                for (auto rit = rhs.degree_ranges[rhs_d]; rit != rhs.degree_ranges[rhs_d + 1]; ++rit) {
                    auto val = op(litraits::value(lit) * ritraits::value(rit));
                    for (auto prd : product(lit, rit)) {
                        result[prd.first] += prd.second*val;
                    }
                }
            }
        }
    }
}

} // namespace dtl



} // namespace algebra
} // namespace esig


#endif//ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_
