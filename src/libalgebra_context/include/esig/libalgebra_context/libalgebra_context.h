//
// Created by user on 05/04/2022.
//

#ifndef ESIG_PATHS_SRC_LIBALGEBRA_CONTEXT_INCLUDE_ESIG_LIBALGEBRA_CONTEXT_LIBALGEBRA_CONTEXT_H_
#define ESIG_PATHS_SRC_LIBALGEBRA_CONTEXT_INCLUDE_ESIG_LIBALGEBRA_CONTEXT_LIBALGEBRA_CONTEXT_H_

#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/context.h>
#include <esig/implementation_types.h>

#include "vector_construction_data.h"
#include <libalgebra/coefficients/coefficients.h>
#include <libalgebra/hall_set.h>
#include <libalgebra/libalgebra.h>
#include <mutex>

namespace esig {
namespace algebra {

namespace dtl {

template<template<typename, typename, typename...> class Vector>
struct vector_type_helper {
    static constexpr vector_type value = vector_type::sparse;
};

template<>
struct vector_type_helper<alg::vectors::dense_vector> {
    static constexpr vector_type value = vector_type::dense;
};


}// namespace dtl


template<deg_t Width, deg_t Depth, typename Coeffs, template<typename, typename, typename...> class Vector, typename... Args>
struct algebra_info<alg::free_tensor<Coeffs, Width, Depth, Vector, Args...>> {
    using tensor_t = alg::free_tensor<Coeffs, Width, Depth, Vector, Args...>;
    using scalar_type = typename Coeffs::S;
    static constexpr coefficient_type ctype() noexcept
    {
        return esig::dtl::get_coeff_type(Coeffs::zero);
    }
    static constexpr vector_type vtype() noexcept
    {
        return dtl::vector_type_helper<Vector>::value;
    }
    static deg_t width(const tensor_t &) noexcept { return Width; }
    static deg_t max_depth(const tensor_t &) noexcept { return Depth; }

    using key_type = typename tensor_t::KEY;
    static key_type convert_key(esig::key_type key)
    {
        return tensor_t::basis.index_to_key(key);
    }
};

template<deg_t Width, deg_t Depth, typename Coeffs, template<typename, typename, typename...> class Vector, typename... Args>
struct algebra_info<alg::lie<Coeffs, Width, Depth, Vector, Args...>> {
    using lie_t = alg::lie<Coeffs, Width, Depth, Vector, Args...>;
    using scalar_type = typename Coeffs::S;
    static constexpr coefficient_type ctype() noexcept
    {
        return esig::dtl::get_coeff_type(Coeffs::zero);
    }
    static constexpr vector_type vtype() noexcept
    {
        return dtl::vector_type_helper<Vector>::value;
    }
    static deg_t width(const lie_t &) noexcept { return Width; }
    static deg_t depth(const lie_t &) noexcept { return Depth; }
    using key_type = typename lie_t::KEY;
    static key_type convert_key(esig::key_type key)
    {
        // For now, the key_type and usage are equivalent between libalgebra and esig::algebra
        return key;
    }
};


template<typename Basis, typename Coeffs, typename Multiplication, template<typename, typename, typename...> class Vector, typename... Args>
struct algebra_info<alg::algebra<Basis, Coeffs, Multiplication, Vector, Args...>> {
    using alg_t = alg::algebra<Basis, Coeffs, Multiplication, Vector, Args...>;
    using scalar_type = typename Coeffs::S;
    static constexpr coefficient_type ctype() noexcept
    {
        return esig::dtl::get_coeff_type(Coeffs::zero);
    }
    static constexpr vector_type vtype() noexcept
    {
        return dtl::vector_type_helper<Vector>::value;
    }
    static deg_t width(const alg_t &) noexcept { return 0; }
    static deg_t depth(const alg_t &) noexcept { return 0; }
    using key_type = typename alg_t::KEY;
    static key_type convert_key(esig::key_type key)
    {
        // For now, the key_type and usage are equivalent between libalgebra and esig::algebra
        return key;
    }
};





namespace dtl {

template <typename ValueType>
struct iterator_helper<alg::vectors::iterators::vector_iterator<ValueType>>
{
    using iter_t = alg::vectors::iterators::vector_iterator<ValueType>;

    static void advance(iter_t& iter) { ++iter; }
    static key_type key(iter_t& iter) { return iter->key(); }
    static const void* value(iter_t& iter) { return &iter->value(); }
    static bool equals(const iter_t& iter1, const iter_t& iter2)
    { return iter1 == iter2; }
};




template<coefficient_type>
struct ctype_helper;

template<>
struct ctype_helper<coefficient_type::dp_real> {
    using type = alg::coefficients::double_field;
};

template<>
struct ctype_helper<coefficient_type::sp_real> {
    using type = alg::coefficients::float_field;
};

template<vector_type>
struct vtype_helper;

template<>
struct vtype_helper<vector_type::dense> {
    template<typename B, typename C, typename...>
    using type = alg::vectors::dense_vector<B, C>;
};

template<>
struct vtype_helper<vector_type::sparse> {
    template<typename B, typename C, typename... Args>
    using type = alg::vectors::sparse_vector<B, C, Args...>;
};

template<deg_t Width, deg_t Depth, coefficient_type CType, vector_type VType>
free_tensor make_free_tensor(const vector_construction_data &data)
{
    using scalar_t = typename ctype_helper<CType>::type::S;

    using vec_t = alg::free_tensor<
            typename ctype_helper<CType>::type,
            Width,
            Depth,
            vtype_helper<VType>::template type>;

    if (data.input_type() == esig::algebra::input_data_type::value_array) {
        return free_tensor(vec_t(
                reinterpret_cast<const scalar_t *>(data.begin()),
                reinterpret_cast<const scalar_t *>(data.end())));
    } else {
        return free_tensor(vec_t());
    }
}

template<deg_t Width, deg_t Depth, coefficient_type CType, vector_type VType>
lie make_lie(const vector_construction_data &data)
{
    using scalar_t = typename ctype_helper<CType>::type::S;

    using vec_t = alg::lie<
            typename ctype_helper<CType>::type,
            Width,
            Depth,
            vtype_helper<VType>::template type>;

    if (data.input_type() == esig::algebra::input_data_type::value_array) {
        return lie(vec_t(
                reinterpret_cast<const scalar_t *>(data.begin()),
                reinterpret_cast<const scalar_t *>(data.end())));
    } else {
        return lie(vec_t());
    }
}


}// namespace dtl


template<typename Lie>
struct to_libalgebra_lie_helper {

    using scalar_type = typename Lie::SCALAR;

    Lie operator()(const char* b, const char* e) const
    {
        return Lie(reinterpret_cast<const scalar_type*>(b), reinterpret_cast<const scalar_type*>(e));
    }

    Lie operator()(data_iterator* data) const
    {
        return Lie();
    }

};


template<deg_t Width, deg_t Depth>
class libalgebra_context : public context
{
    static const alg::lie_basis<Width, Depth> m_lie_basis;
    static const alg::free_tensor_basis<Width, Depth> m_tensor_basis;

    template<coefficient_type CType, vector_type VType>
    static free_tensor lie_to_tensor_impl(const lie &arg)
    {
        using tensor_t = alg::free_tensor<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        alg::maps<typename dtl::ctype_helper<CType>::type, Width, Depth, tensor_t, lie_t> maps;
        return free_tensor(maps.l2t(algebra_base_access::template get<lie_t>(arg)));
    }

    template<coefficient_type CType, vector_type VType>
    static lie tensor_to_lie_impl(const free_tensor &arg)
    {
        using tensor_t = alg::free_tensor<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        alg::maps<typename dtl::ctype_helper<CType>::type, Width, Depth, tensor_t, lie_t> maps;
        return lie(maps.t2l(algebra_base_access::template get<tensor_t>(arg)));
    }

    template<coefficient_type CType, vector_type VType>
    static lie cbh_impl(const std::vector<lie> &lies)
    {
        using tensor_t = alg::free_tensor<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        alg::cbh<typename dtl::ctype_helper<CType>::type, Width, Depth, tensor_t, lie_t> cbh;
        std::vector<const lie_t *> real_lies;
        real_lies.reserve(lies.size());
        for (auto &lw : lies) {
            real_lies.push_back(&algebra_base_access::template get<lie_t>(lw));
        }
        return lie(cbh.full(real_lies));
    }

    template<typename Lie, typename Tensor>
    static Tensor compute_signature(signature_data data)
    {
        to_libalgebra_lie_helper<Lie> helper;
        Tensor result(typename Tensor::SCALAR(1));
        alg::maps<typename Tensor::coefficient_field, Width, Depth, Tensor, Lie> maps;
        for (auto incr : data.template iter_increments(helper)) {
            result.fmexp_inplace(maps.l2t(incr));
        }
        return result;
    }

    template<coefficient_type CType, vector_type VType>
    static free_tensor signature_impl(signature_data data)
    {
        using tensor_t = alg::free_tensor<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        return free_tensor(compute_signature<lie_t, tensor_t>(std::move(data)));
    }

    template<coefficient_type CType, vector_type VType>
    static lie log_signature_impl(signature_data data)
    {
        using tensor_t = alg::free_tensor<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        alg::maps<typename dtl::ctype_helper<CType>::type, Width, Depth, tensor_t, lie_t> maps;
        return lie(maps.t2l(log(compute_signature<lie_t, tensor_t>(std::move(data)))));
    }

    template<typename Lie>
    static Lie ad_x_n(deg_t d, const Lie &x, const Lie &y)
    {
        auto tmp = x * y;
        while (--d) {
            tmp = x * tmp;
        }
        return tmp;
    }

    template<typename Lie>
    static Lie derive_series_compute(const Lie &incr, const Lie &perturbation)
    {
        Lie result;
        typename Lie::SCALAR factor(1);
        for (deg_t d = 1; d <= Depth; ++d) {
            factor /= static_cast<typename Lie::SCALAR>(d + 1);
            if (d & 1) {
                result.add_scal_div(ad_x_n(d, incr, perturbation), factor);
            } else {
                result.sub_scal_div(ad_x_n(d, incr, perturbation), factor);
            }
        }
        return result;
    }

    template<typename Lie, typename Tensor>
    static Tensor compute_single_deriv(const Lie &incr, const Tensor &sig, const Lie &perturb)
    {
        alg::maps<typename Lie::coefficient_field, Width, Depth, Tensor, Lie> maps;
        return sig * maps.l2t(derive_series_compute(incr, perturb));
    }

    template<coefficient_type CType, vector_type VType>
    static free_tensor sig_derivative_impl(const std::vector<derivative_compute_info> &info)
    {
        using tensor_t = alg::free_tensor<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                typename dtl::ctype_helper<CType>::type,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;


        alg::maps<typename lie_t::coefficient_field, Width, Depth, tensor_t, lie_t> maps;
        tensor_t result;

        for (const auto &data : info) {
            const auto &perturb = algebra_base_access::get<lie_t>(data.perturbation);
            const auto &incr = algebra_base_access::get<lie_t>(data.logsig_of_interval);
            auto sig = exp(maps.l2t(incr));

            result *= sig;
            result += compute_single_deriv(incr, sig, perturb);
        }

        return free_tensor(std::move(result));
    }

public:
    deg_t width() const noexcept override
    {
        return Width;
    }
    deg_t depth() const noexcept override
    {
        return Depth;
    }
    dimn_t lie_size(deg_t d) const noexcept override
    {
        if (d == 0) {
            return 0;
        } else {
            return m_lie_basis.start_of_degree(d + 1);
        }
    }
    dimn_t tensor_size(deg_t d) const noexcept override
    {
        if (d == 0) {
            return 1;
        } else {
            return m_tensor_basis.start_of_degree(d + 1);
        }
    }
    free_tensor construct_tensor(const vector_construction_data &data) const override
    {
#define ESIG_SWITCH_FN(CTYPE, VTYPE) dtl::make_free_tensor<Width, Depth, CTYPE, VTYPE>(data)
            ESIG_MAKE_SWITCH(data.ctype(), data.vtype())
#undef ESIG_SWITCH_FN
    }
    lie construct_lie(const vector_construction_data &data) const override
    {
#define ESIG_SWITCH_FN(CTYPE, VTYPE) dtl::make_lie<Width, Depth, CTYPE, VTYPE>(data)
            ESIG_MAKE_SWITCH(data.ctype(), data.vtype())
#undef ESIG_SWITCH_FN
    }
    free_tensor lie_to_tensor(const lie &arg) const override
    {
#define ESIG_SWITCH_FN(CTYPE, VTYPE) lie_to_tensor_impl<CTYPE, VTYPE>(arg)
            ESIG_MAKE_SWITCH(arg.coeff_type(), arg.storage_type())
#undef ESIG_SWITCH_FN
    }
    lie tensor_to_lie(const free_tensor &arg) const override
    {
#define ESIG_SWITCH_FN(CTYPE, VTYPE) tensor_to_lie_impl<CTYPE, VTYPE>(arg)
            ESIG_MAKE_SWITCH(arg.coeff_type(), arg.storage_type())
#undef ESIG_SWITCH_FN
    }
    lie cbh(const std::vector<lie> &lies, coefficient_type ctype, vector_type vtype) const override
    {
#define ESIG_SWITCH_FN(CTYPE, VTYPE) cbh_impl<CTYPE, VTYPE>(lies)
            ESIG_MAKE_SWITCH(ctype, vtype)
#undef ESIG_SWITCH_FN
    }
    free_tensor to_signature(const lie &log_sig) const override
    {
        return lie_to_tensor(log_sig).exp();
    }
    free_tensor signature(signature_data data) const override
    {
#define ESIG_SWITCH_FN(CTYPE, VTYPE) signature_impl<CTYPE, VTYPE>(std::move(data))
            ESIG_MAKE_SWITCH(data.ctype(), data.vtype())
#undef ESIG_SWITCH_FN
    }
    lie log_signature(signature_data data) const override
    {
#define ESIG_SWITCH_FN(CTYPE, VTYPE) log_signature_impl<CTYPE, VTYPE>(std::move(data))
            ESIG_MAKE_SWITCH(data.ctype(), data.vtype())
#undef ESIG_SWITCH_FN
    }
    free_tensor sig_derivative(const std::vector<derivative_compute_info> &info, coefficient_type ctype, vector_type vtype, vector_type type) const override
    {
        if (info.empty()) {
            return zero_tensor(ctype, vtype);
        }
#define ESIG_SWITCH_FN(CTYPE, VTYPE) sig_derivative_impl<CTYPE, VTYPE>(info)
        ESIG_MAKE_SWITCH(ctype, vtype)
#undef ESIG_SWITCH_FN
    }
};


template<deg_t Width, deg_t Depth>
const alg::lie_basis<Width, Depth> libalgebra_context<Width, Depth>::m_lie_basis;

template<deg_t Width, deg_t Depth>
const alg::free_tensor_basis<Width, Depth> libalgebra_context<Width, Depth>::m_tensor_basis;



namespace dtl {

struct pair_hash {
    using argument_type = std::pair<deg_t, deg_t>;
    using result_type = std::size_t;

    inline result_type operator()(const argument_type &arg) const noexcept
    {
        static constexpr deg_t shift = std::numeric_limits<std::size_t>::digits / 2;
        std::size_t result = static_cast<std::size_t>(arg.first) << shift;
        result += static_cast<std::size_t>(arg.second);
        return result;
    }
};

using la_config = std::pair<esig::deg_t, esig::deg_t>;
using context_map = std::unordered_map<la_config, std::shared_ptr<context>, pair_hash>;



} // namespace dtl

void install_default_libalgebra_contexts();



}// namespace algebra
}// namespace esig

#endif//ESIG_PATHS_SRC_LIBALGEBRA_CONTEXT_INCLUDE_ESIG_LIBALGEBRA_CONTEXT_LIBALGEBRA_CONTEXT_H_
