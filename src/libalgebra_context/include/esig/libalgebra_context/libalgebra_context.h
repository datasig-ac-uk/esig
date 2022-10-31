//
// Created by user on 05/04/2022.
//

#ifndef ESIG_PATHS_SRC_LIBALGEBRA_CONTEXT_INCLUDE_ESIG_LIBALGEBRA_CONTEXT_LIBALGEBRA_CONTEXT_H_
#define ESIG_PATHS_SRC_LIBALGEBRA_CONTEXT_INCLUDE_ESIG_LIBALGEBRA_CONTEXT_LIBALGEBRA_CONTEXT_H_

#include <esig/implementation_types.h>

#include <mutex>

#include <boost/functional/hash.hpp>

#include <libalgebra/hall_set.h>
#include <libalgebra/libalgebra.h>
#include <libalgebra/coefficients/coefficients.h>

#include <boost/move/utility_core.hpp>
#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/context.h>

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


template <deg_t Width, deg_t Depth>
class basis_implementation<alg::free_tensor_basis<Width, Depth>>
    : public basis_interface
{
    using basis_type = alg::free_tensor_basis<Width, Depth>;
    const basis_type& m_basis;

public:

    explicit basis_implementation(const basis_type& basis) : m_basis(basis)
    {}

    boost::optional<deg_t> width() const noexcept override {
        return Width;
    }
    boost::optional<deg_t> depth() const noexcept override {
        return Depth;
    }
    boost::optional<deg_t> degree(key_type key) const noexcept override {
        return m_basis.degree(m_basis.index_to_key(key));
    }
    std::string key_to_string(key_type key) const noexcept override {
        return m_basis.key2string(m_basis.index_to_key(key));
    }
    dimn_t size(int i) const noexcept override {
        return m_basis.start_of_degree(unsigned(i+1));
    }
    dimn_t start_of_degree(int i) const noexcept override {
        return m_basis.start_of_degree(unsigned(i));
    }
    boost::optional<key_type> lparent(key_type key) const noexcept override {
        return m_basis.key_to_index(m_basis.lparent(m_basis.index_to_key(key)));
    }
    boost::optional<key_type> rparent(key_type key) const noexcept override {
        return m_basis.key_to_index(m_basis.rparent(m_basis.index_to_key(key)));
    }
    key_type index_to_key(dimn_t idx) const noexcept override {
        return idx;
    }
    dimn_t key_to_index(key_type key) const noexcept override {
        return key;
    }
    let_t first_letter(key_type key) const noexcept override {
        return m_basis.getletter(lparent(key));
    }
    key_type key_of_letter(let_t letter) const noexcept override {
        return m_basis.key_to_index(m_basis.keyofletter(letter));
    }
    bool letter(key_type key) const noexcept override {
        return m_basis.letter(m_basis.index_to_key(key));
    }
};


template <deg_t Width, deg_t Depth>
class basis_implementation<alg::lie_basis<Width, Depth>>
    : public basis_interface
{
    using basis_type = alg::lie_basis<Width, Depth>;
    const basis_type& m_basis;
public:

    explicit basis_implementation(const basis_type& basis) : m_basis(basis)
    {}


    boost::optional<deg_t> width() const noexcept override {
        return Width;
    }
    boost::optional<deg_t> depth() const noexcept override {
        return Depth;
    }
    boost::optional<deg_t> degree(key_type key) const noexcept override {
        return m_basis.degree(key);
    }
    std::string key_to_string(key_type key) const noexcept override {
        return m_basis.key2string(key);
    }
    dimn_t size(int i) const noexcept override {
        return m_basis.start_of_degree(unsigned(i+1));
    }
    dimn_t start_of_degree(int i) const noexcept override {
        return m_basis.start_of_degree(unsigned(i));
    }
    boost::optional<key_type> lparent(key_type key) const noexcept override {
        return m_basis.lparent(key);
    }
    boost::optional<key_type> rparent(key_type key) const noexcept override {
        return m_basis.rparent(key);
    }
    key_type index_to_key(dimn_t idx) const noexcept override {
        return idx;
    }
    dimn_t key_to_index(key_type key) const noexcept override {
        return key;
    }
    let_t first_letter(key_type key) const noexcept override {
        return m_basis.getletter(lparent(key));
    }
    key_type key_of_letter(let_t letter) const noexcept override {
        return m_basis.keyofletter(letter);
    }

    bool letter(key_type key) const noexcept override {
        return m_basis.letter(key);
    }
};

}// namespace dtl


template<deg_t Width, deg_t Depth, typename Coeffs, template<typename, typename, typename...> class Vector, typename... Args>
struct algebra_info<alg::free_tensor<Coeffs, Width, Depth, Vector, Args...>> {
    using tensor_t = alg::free_tensor<Coeffs, Width, Depth, Vector, Args...>;
    using scalar_type = typename Coeffs::S;
    using rational_type = typename Coeffs::Q;
    using reference = decltype(std::declval<tensor_t>()[std::declval<typename tensor_t::KEY>()]);
    using const_reference = decltype(std::declval<const tensor_t>()[std::declval<typename tensor_t::KEY>()]);
    using pointer = scalar_type*;
    using const_pointer = const scalar_type*;

    static constexpr coefficient_type ctype() noexcept
    { return dtl::get_coeff_type(Coeffs::zero); }
    static constexpr vector_type vtype() noexcept
    { return dtl::vector_type_helper<Vector>::value; }
    static deg_t width(const tensor_t *) noexcept { return Width; }
    static deg_t max_depth(const tensor_t *) noexcept { return Depth; }

    using this_key_type = typename tensor_t::KEY;
    static this_key_type convert_key(const tensor_t*, esig::key_type key)
    { return tensor_t::basis.index_to_key(key); }

    static deg_t degree(const tensor_t* inst, key_type key) noexcept
    { return native_degree(inst, convert_key(inst, key)); }
    static deg_t native_degree(const tensor_t*, this_key_type key) noexcept
    { return tensor_t::basis.degree(key); }

    static key_type first_key(const tensor_t*) noexcept
    { return 0; }
    static key_type last_key(const tensor_t*) noexcept
    { return tensor_t::basis.size(); }

    using key_prod_container = boost::container::small_vector<std::pair<key_type, int>, 1>;
    static key_prod_container key_product(const tensor_t* inst, key_type k1, key_type k2)
    {
        return {tensor_t::basis.key_to_index(convert_key(inst, k1)*convert_key(inst, k2)), 1};
    }

    static tensor_t create_like(const tensor_t& instance) { return tensor_t(); }
};

template<deg_t Width, deg_t Depth, typename Coeffs, template<typename, typename, typename...> class Vector, typename... Args>
struct algebra_info<alg::lie<Coeffs, Width, Depth, Vector, Args...>> {
    using lie_t = alg::lie<Coeffs, Width, Depth, Vector, Args...>;

    using scalar_type = typename Coeffs::S;
    using rational_type = typename Coeffs::Q;
    using reference = decltype(std::declval<lie_t>()[std::declval<typename lie_t::KEY>()]);
    using const_reference = decltype(std::declval<const lie_t>()[std::declval<typename lie_t::KEY>()]);
    using pointer = scalar_type*;
    using const_pointer = const scalar_type*;

    static constexpr coefficient_type ctype() noexcept
    {
        return dtl::get_coeff_type(Coeffs::zero);
    }
    static constexpr vector_type vtype() noexcept
    {
        return dtl::vector_type_helper<Vector>::value;
    }
    static deg_t width(const lie_t *) noexcept { return Width; }
    static deg_t max_depth(const lie_t *) noexcept { return Depth; }

    using this_key_type = typename lie_t::KEY;
    static this_key_type convert_key(const lie_t*, esig::key_type key)
    {
        // For now, the key_type and usage are equivalent between libalgebra and esig::algebra
        return key;
    }



};


template<typename Basis, typename Coeffs, typename Multiplication, template<typename, typename, typename...> class Vector, typename... Args>
struct algebra_info<alg::algebra<Basis, Coeffs, Multiplication, Vector, Args...>> {
    using alg_t = alg::algebra<Basis, Coeffs, Multiplication, Vector, Args...>;

    using scalar_type = typename Coeffs::S;
    using rational_type = typename Coeffs::Q;
    using reference = decltype(std::declval<alg_t>()[std::declval<typename alg_t::KEY>()]);
    using const_reference = decltype(std::declval<const alg_t>()[std::declval<typename alg_t::KEY>()]);
    using pointer = scalar_type*;
    using const_pointer = const scalar_type*;

    static constexpr coefficient_type ctype() noexcept
    {
        return dtl::get_coeff_type(Coeffs::zero);
    }
    static constexpr vector_type vtype() noexcept
    {
        return dtl::vector_type_helper<Vector>::value;
    }
    static deg_t width(const alg_t* ) noexcept { return 0; }
    static deg_t max_depth(const alg_t*) noexcept { return 0; }
    using this_key_type = typename alg_t::KEY;
    static this_key_type convert_key(esig::key_type key)
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


template<deg_t Width, deg_t Depth, coefficient_type CType>
class libalgebra_context : public context
{
    using coeff_ring = typename dtl::ctype_helper<CType>::type;
    static const alg::lie_basis<Width, Depth> m_lie_basis;
    static const alg::free_tensor_basis<Width, Depth> m_tensor_basis;


    template<vector_type VType>
    free_tensor make_free_tensor(const vector_construction_data &data) const
    {
        using scalar_t = typename coeff_ring::S;

        using vec_t = alg::free_tensor<
            coeff_ring,
            Width,
            Depth,
            dtl::vtype_helper<VType>::template type>;

        if (data.input_type() == esig::algebra::input_data_type::value_array) {
            return free_tensor(vec_t(
                reinterpret_cast<const scalar_t *>(data.begin()),
                reinterpret_cast<const scalar_t *>(data.end())), this);
        } else {
            return free_tensor(vec_t(), this);
        }
    }

    template<vector_type VType>
    lie make_lie(const vector_construction_data &data) const
    {
        using scalar_t = typename coeff_ring::S;

        using vec_t = alg::lie<
            coeff_ring,
            Width,
            Depth,
            dtl::vtype_helper<VType>::template type>;

        if (data.input_type() == esig::algebra::input_data_type::value_array) {
            return lie(vec_t(
                reinterpret_cast<const scalar_t *>(data.begin()),
                reinterpret_cast<const scalar_t *>(data.end())), this);
        } else {
            return lie(vec_t(), this);
        }
    }

    template< vector_type VType>
    free_tensor lie_to_tensor_impl(const lie &arg) const
    {
        using tensor_t = alg::free_tensor<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        alg::maps<coeff_ring, Width, Depth, tensor_t, lie_t> maps;
        return free_tensor(maps.l2t(algebra_access<lie_interface>::template get<lie_t>(arg)), this);
    }

    template<vector_type VType>
    lie tensor_to_lie_impl(const free_tensor &arg) const
    {
        using tensor_t = alg::free_tensor<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        alg::maps<coeff_ring, Width, Depth, tensor_t, lie_t> maps;
        return lie(maps.t2l(algebra_access<free_tensor_interface>::template get<tensor_t>(arg)), this);
    }

    template<vector_type VType>
    lie cbh_impl(const std::vector<lie> &lies) const
    {
        using tensor_t = alg::free_tensor<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        alg::cbh<coeff_ring, Width, Depth, tensor_t, lie_t> cbh;
        std::vector<const lie_t *> real_lies;
        real_lies.reserve(lies.size());
        for (auto &lw : lies) {
            real_lies.push_back(&algebra_access<lie_interface>::template get<lie_t>(lw));
        }
        return lie(cbh.full(real_lies), this);
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

    template<vector_type VType>
    free_tensor signature_impl(signature_data data) const
    {
        using tensor_t = alg::free_tensor<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        return free_tensor(compute_signature<lie_t, tensor_t>(std::move(data)), this);
    }

    template<vector_type VType>
    lie log_signature_impl(signature_data data) const
    {
        using tensor_t = alg::free_tensor<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        alg::maps<coeff_ring, Width, Depth, tensor_t, lie_t> maps;
        return lie(maps.t2l(log(compute_signature<lie_t, tensor_t>(std::move(data)))), this);
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
        alg::maps<coeff_ring, Width, Depth, Tensor, Lie> maps;
        return sig * maps.l2t(derive_series_compute(incr, perturb));
    }

    template<vector_type VType>
    free_tensor sig_derivative_impl(const std::vector<derivative_compute_info> &info) const
    {
        using tensor_t = alg::free_tensor<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;
        using lie_t = alg::lie<
                coeff_ring,
                Width,
                Depth,
                dtl::vtype_helper<VType>::template type>;


        alg::maps<coeff_ring, Width, Depth, tensor_t, lie_t> maps;
        tensor_t result;

        for (const auto &data : info) {
            const auto &perturb = algebra_access<lie_interface>::template
                get<lie_t>(data.perturbation);
            const auto &incr = algebra_access<lie_interface>::template
                get<lie_t>(data.logsig_of_interval);
            auto sig = exp(maps.l2t(incr));

            result *= sig;
            result += compute_single_deriv(incr, sig, perturb);
        }

        return free_tensor(std::move(result), this);
    }

public:
    coefficient_type ctype() const noexcept override {
        return CType;
    }
    std::shared_ptr<const context> get_alike(deg_t new_depth) const override {
        return std::shared_ptr<const context>();
    }
    std::shared_ptr<const context> get_alike(coefficient_type new_coeff) const override {
        return std::shared_ptr<const context>();
    }
    std::shared_ptr<const context> get_alike(deg_t new_depth, coefficient_type new_coeff) const override {
        return std::shared_ptr<const context>();
    }
    std::shared_ptr<const context> get_alike(deg_t new_width, deg_t new_depth, coefficient_type new_coeff) const override {
        return std::shared_ptr<const context>();
    }
    std::shared_ptr<data_allocator> coefficient_alloc() const override {
        return std::shared_ptr<data_allocator>();
    }
    std::shared_ptr<data_allocator> pair_alloc() const override {
        return std::shared_ptr<data_allocator>();
    }
    free_tensor convert(const free_tensor &arg, vector_type new_vec_type) const override {
        return free_tensor();
    }
    lie convert(const lie &arg, vector_type new_vec_type) const override {
        return lie();
    }
//    const algebra_basis &borrow_tbasis() const noexcept override {
//        return ;
//    }
//    const algebra_basis &borrow_lbasis() const noexcept override {
//        return <#initializer #>;
//    }
    basis get_tensor_basis() const override {
        return basis(m_tensor_basis);
    }
    basis get_lie_basis() const override {
        return basis(m_lie_basis);
    }

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
#define ESIG_SWITCH_FN(VTYPE) make_free_tensor<VTYPE>(data)
            ESIG_MAKE_VTYPE_SWITCH(data.vtype())
#undef ESIG_SWITCH_FN
    }
    lie construct_lie(const vector_construction_data &data) const override
    {
#define ESIG_SWITCH_FN(VTYPE) make_lie<VTYPE>(data)
            ESIG_MAKE_VTYPE_SWITCH(data.vtype())
#undef ESIG_SWITCH_FN
    }
    free_tensor lie_to_tensor(const lie &arg) const override
    {
#define ESIG_SWITCH_FN(VTYPE) lie_to_tensor_impl<VTYPE>(arg)
            ESIG_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef ESIG_SWITCH_FN
    }
    lie tensor_to_lie(const free_tensor &arg) const override
    {
#define ESIG_SWITCH_FN(VTYPE) tensor_to_lie_impl<VTYPE>(arg)
            ESIG_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef ESIG_SWITCH_FN
    }
    lie cbh(const std::vector<lie> &lies, vector_type vtype) const override
    {
#define ESIG_SWITCH_FN(VTYPE) cbh_impl<VTYPE>(lies)
            ESIG_MAKE_VTYPE_SWITCH(vtype)
#undef ESIG_SWITCH_FN
    }
    free_tensor to_signature(const lie &log_sig) const override
    {
        return lie_to_tensor(log_sig).exp();
    }
    free_tensor signature(signature_data data) const override
    {
#define ESIG_SWITCH_FN(VTYPE) signature_impl<VTYPE>(std::move(data))
            ESIG_MAKE_VTYPE_SWITCH(data.vtype())
#undef ESIG_SWITCH_FN
    }
    lie log_signature(signature_data data) const override
    {
#define ESIG_SWITCH_FN(VTYPE) log_signature_impl<VTYPE>(std::move(data))
            ESIG_MAKE_VTYPE_SWITCH(data.vtype())
#undef ESIG_SWITCH_FN
    }
    free_tensor sig_derivative(const std::vector<derivative_compute_info> &info, vector_type vtype, vector_type type) const override
    {
        if (info.empty()) {
            return zero_tensor(vtype);
        }
#define ESIG_SWITCH_FN(VTYPE) sig_derivative_impl<VTYPE>(info)
        ESIG_MAKE_VTYPE_SWITCH(vtype)
#undef ESIG_SWITCH_FN
    }
};


template<deg_t Width, deg_t Depth, coefficient_type CType>
const alg::lie_basis<Width, Depth> libalgebra_context<Width, Depth, CType>::m_lie_basis;

template<deg_t Width, deg_t Depth, coefficient_type CType>
const alg::free_tensor_basis<Width, Depth> libalgebra_context<Width, Depth, CType>::m_tensor_basis;



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

using la_config = std::tuple<esig::deg_t, esig::deg_t, coefficient_type>;
using context_map = std::unordered_map<la_config, std::shared_ptr<const context>, boost::hash<la_config>>;



} // namespace dtl

void install_default_libalgebra_contexts();



}// namespace algebra
}// namespace esig

#endif//ESIG_PATHS_SRC_LIBALGEBRA_CONTEXT_INCLUDE_ESIG_LIBALGEBRA_CONTEXT_LIBALGEBRA_CONTEXT_H_
