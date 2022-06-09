//
// Created by user on 19/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_
#define ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_

#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/coefficients.h>

#include <cassert>
#include <typeinfo>
#include <iostream>


namespace esig {
namespace algebra {


enum class vector_type
{
    sparse,
    dense
};

template <typename Algebra>
struct algebra_info
{
    using scalar_type = typename Algebra::scalar_type;

    static constexpr coefficient_type ctype() noexcept;
    static constexpr vector_type vtype() noexcept;
    static deg_t width(const Algebra& instance) noexcept;
    static deg_t max_depth(const Algebra& instance) noexcept;

    using this_key_type = key_type;
    static this_key_type convert_key(esig::key_type key) noexcept;
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
deg_t algebra_info<Algebra>::width(const Algebra& instance) noexcept
{
    return instance.width();
}
template<typename Algebra>
deg_t algebra_info<Algebra>::max_depth(const Algebra& instance) noexcept
{
    return instance.depth();
}

template<typename Algebra>
key_type algebra_info<Algebra>::convert_key(esig::key_type key) noexcept
{
    return key;
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



} // namespace algebra
} // namespace esig


#endif//ESIG_PATHS_SRC_ALGEBRA_TMP_INCLUDE_ESIG_ALGEBRA_ALGEBRA_TRAITS_H_
