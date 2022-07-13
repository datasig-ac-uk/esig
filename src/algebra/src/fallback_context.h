//
// Created by user on 20/03/2022.
//

#ifndef ESIG_PATHS_SRC_CONTEXT_CONTEXT_CPP_FALLBACK_CONTEXT1_H_
#define ESIG_PATHS_SRC_CONTEXT_CONTEXT_CPP_FALLBACK_CONTEXT1_H_

#include <esig/algebra/context.h>
#include "dense_tensor.h"
#include "dense_lie.h"
#include "sparse_tensor.h"
#include "sparse_lie.h"
#include "lie_basis/lie_basis.h"
#include "lie_basis/hall_extension.h"
#include "fallback_maps.h"
#include "detail/converting_lie_iterator_adaptor.h"
#include <memory>
#include <unordered_map>
#include <bits/shared_ptr.h>


namespace esig {
namespace algebra {

namespace dtl {

template<coefficient_type, vector_type>
struct vector_selector;

} // namespace dtl

template <coefficient_type CType, vector_type VType>
using ftensor_type = typename dtl::vector_selector<CType, VType>::free_tensor;

template <coefficient_type CType, vector_type VType>
using lie_type = typename dtl::vector_selector<CType, VType>::lie;

template<typename Lie>
struct to_lie_helper {
    using scalar_type = typename Lie::scalar_type;

    to_lie_helper(std::shared_ptr<lie_basis> b, deg_t degree)
        : m_basis(std::move(b)), m_degree(degree) {}

    Lie operator()(const char *b, const char *e) const {
        return {m_basis,
                reinterpret_cast<const scalar_type *>(b),
                reinterpret_cast<const scalar_type *>(e)};
    }

    Lie operator()(data_iterator *data) const {
        Lie result(m_basis);
        while (data->next_sparse()) {
            auto kv = reinterpret_cast<const std::pair<key_type, scalar_type> *>(data->sparse_kv_pair());
            result[kv->first] = kv->second;
        }
        return result;
    }

private:
    std::shared_ptr<lie_basis> m_basis;
    deg_t m_degree;
};




class fallback_context
    : public context
{
    deg_t m_width;
    deg_t m_depth;
    coefficient_type m_ctype;
    std::shared_ptr<tensor_basis> m_tensor_basis;
    std::shared_ptr<lie_basis> m_lie_basis;

    std::shared_ptr<data_allocator> m_coeff_alloc;
    std::shared_ptr<data_allocator> m_pair_alloc;

    maps m_maps;

    template <coefficient_type CType, vector_type VType>
    lie_type<CType, VType> tensor_to_lie_impl(const ftensor_type<CType, VType>& arg) const;

    template <coefficient_type CType, vector_type VType>
    ftensor_type<CType, VType> lie_to_tensor_impl(const lie_type<CType, VType>& arg) const;

    template <coefficient_type CType, vector_type VType>
    lie_type<CType, VType> cbh_impl(const std::vector<lie>& lies) const;

    template<coefficient_type CType, vector_type VType>
    ftensor_type<CType, VType> compute_signature(signature_data data) const;

    template<typename Lie>
    Lie ad_x_n(deg_t d, const Lie &x, const Lie &y) const;

    template<typename Lie>
    Lie derive_series_compute(const Lie &incr, const Lie &pert) const;

    template<coefficient_type CType, vector_type VType>
    ftensor_type<CType, VType> single_sig_derivative(
        const lie_type<CType, VType> &incr,
        const ftensor_type<CType, VType> &sig,
        const lie_type<CType, VType> &perturb
        ) const;

    template<coefficient_type CType, vector_type VType>
    ftensor_type<CType, VType>
    sig_derivative_impl(const std::vector<derivative_compute_info> &info) const;
public:

    fallback_context(deg_t width, deg_t depth, coefficient_type ctype);


    deg_t width() const noexcept override;
    deg_t depth() const noexcept override;
    coefficient_type ctype() const noexcept override;

    const algebra_basis &borrow_tbasis() const noexcept override;
    const algebra_basis &borrow_lbasis() const noexcept override;

    std::shared_ptr<context> get_alike(deg_t new_depth) const override;
    std::shared_ptr<context> get_alike(coefficient_type new_coeff) const override;
    std::shared_ptr<context> get_alike(deg_t new_depth, coefficient_type new_coeff) const override;
    std::shared_ptr<context> get_alike(deg_t new_width, deg_t new_depth, coefficient_type new_coeff) const override;

    std::shared_ptr<data_allocator> coefficient_alloc() const override;
    std::shared_ptr<data_allocator> pair_alloc() const override;

    free_tensor zero_tensor(vector_type vtype) const override;
    lie zero_lie(vector_type vtype) const override;
    lie cbh(const std::vector<lie> &lies, vector_type vtype) const override;
    free_tensor sig_derivative(const std::vector<derivative_compute_info> &info, vector_type vtype, vector_type type) const override;
    dimn_t lie_size(deg_t d) const noexcept override;
    dimn_t tensor_size(deg_t d) const noexcept override;
    std::shared_ptr<tensor_basis> get_tensor_basis() const noexcept;
    std::shared_ptr<lie_basis> get_lie_basis() const noexcept;
    free_tensor construct_tensor(const vector_construction_data &data) const override;
    lie construct_lie(const vector_construction_data &data) const override;
    free_tensor lie_to_tensor(const lie &arg) const override;
    lie tensor_to_lie(const free_tensor &arg) const override;
    free_tensor to_signature(const lie &log_sig) const override;
    free_tensor signature(signature_data data) const override;
    lie log_signature(signature_data data) const override;

};


class fallback_context_maker : public context_maker
{
public:
    std::shared_ptr<context> get_context(deg_t width, deg_t depth, coefficient_type ctype) const override;
    bool can_get(deg_t width, deg_t depth, coefficient_type ctype) const noexcept override;
    int get_priority(const std::vector<std::string> &preferences) const noexcept override;
};



namespace dtl {

template<>
struct vector_selector<coefficient_type::sp_real, vector_type::sparse> {
    using free_tensor = sparse_tensor<float>;
    using lie = sparse_lie<float>;
};

template<>
struct vector_selector<coefficient_type::sp_real, vector_type::dense> {
    using free_tensor = dense_tensor<float>;
    using lie = dense_lie<float>;
};

template<>
struct vector_selector<coefficient_type::dp_real, vector_type::sparse> {
    using free_tensor = sparse_tensor<double>;
    using lie = sparse_lie<double>;
};

template<>
struct vector_selector<coefficient_type::dp_real, vector_type::dense> {
    using free_tensor = dense_tensor<double>;
    using lie = dense_lie<double>;
};
} // namespace dtl

template<coefficient_type CType, vector_type VType>
lie_type<CType, VType> fallback_context::cbh_impl(const std::vector<lie> &lies) const {
    using lie_t = lie_type<CType, VType>;
    dtl::converting_lie_iterator_adaptor<lie_t> begin(m_lie_basis, lies.begin()), end(m_lie_basis, lies.end());
    return m_maps.cbh(begin, end);
}

template<coefficient_type CType, vector_type VType>
ftensor_type<CType, VType> fallback_context::lie_to_tensor_impl(const lie_type<CType, VType> &arg) const
{
    return m_maps.template lie_to_tensor<lie_type<CType, VType>, ftensor_type<CType, VType>>(arg);
}

template<coefficient_type CType, vector_type VType>
lie_type<CType, VType> fallback_context::tensor_to_lie_impl(const ftensor_type<CType, VType> &arg) const
{
     return m_maps.template tensor_to_lie<lie_type<CType, VType>, ftensor_type<CType, VType>>(arg);
}

template<coefficient_type CType, vector_type VType>
ftensor_type<CType, VType> fallback_context::compute_signature(signature_data data) const {
    //TODO: In the future, use a method on signature_data to get the
    // correct depth for the increments.
    to_lie_helper<lie_type<CType, VType>> helper(get_lie_basis(), 1);
    ftensor_type<CType, VType> result(get_tensor_basis(), typename ftensor_type<CType, VType>::scalar_type(1));
    for (auto incr : data.template iter_increments(helper)) {
        result.fmexp_inplace(lie_to_tensor_impl<CType, VType>(incr));
    }

    return result;
}

template<typename Lie>
Lie fallback_context::ad_x_n(deg_t d, const Lie &x, const Lie &y) const
{
    auto tmp = x * y;// [x, y]
    while (--d) {
        tmp = x * tmp;// x*tmp is [x, tmp] here
    }
    return tmp;
}

template<typename Lie>
Lie fallback_context::derive_series_compute(const Lie &incr, const Lie &pert) const
{
    Lie result(get_lie_basis());

    typename Lie::scalar_type factor(1);
    for (deg_t d = 1; d <= m_depth; ++d) {
        factor /= static_cast<typename Lie::scalar_type>(d + 1);
        if (d & 1) {
            result.add_scal_div(ad_x_n(d, incr, pert), factor);
        } else {
            result.sub_scal_div(ad_x_n(d, incr, pert), factor);
        }
    }
    return result;
}

template<coefficient_type CType, vector_type VType>
ftensor_type<CType, VType> fallback_context::single_sig_derivative(
    const lie_type<CType, VType> &incr,
    const ftensor_type<CType, VType> &sig,
    const lie_type<CType, VType> &perturb
    ) const
{
    return sig * lie_to_tensor_impl<CType, VType>(derive_series_compute(incr, perturb));
}

template<coefficient_type CType, vector_type VType>
ftensor_type<CType, VType> fallback_context::sig_derivative_impl(const std::vector<derivative_compute_info> &info) const
{
    if (info.empty()) {
        return ftensor_type<CType, VType>(get_tensor_basis());
    }

    ftensor_type<CType, VType> result(get_tensor_basis());

    for (const auto &data : info) {
        const auto &perturb = lie_base_access::get<lie_type<CType, VType>>(data.perturbation);
        const auto &incr = lie_base_access::get<lie_type<CType, VType>>(data.logsig_of_interval);
        auto sig = exp(lie_to_tensor_impl<CType, VType>(incr));

        result *= sig;
        result += single_sig_derivative<CType, VType>(incr, sig, perturb);
    }

    return result;
}


}// namespace algebra
}// namespace esig


#endif//ESIG_PATHS_SRC_CONTEXT_CONTEXT_CPP_FALLBACK_CONTEXT1_H_
