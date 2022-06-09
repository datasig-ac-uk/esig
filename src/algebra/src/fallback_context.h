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

#include <memory>
#include <unordered_map>
#include <bits/shared_ptr.h>


namespace esig {
namespace algebra {

namespace dtl {
    class expand_binop
    {
        std::shared_ptr<tensor_basis> m_basis;
    public:
        using result_t = std::vector<std::pair<key_type, int>>;

        explicit expand_binop(std::shared_ptr<tensor_basis> arg);
        result_t operator()(const result_t& a, const result_t& b) const;
    };


struct bracketing_item : std::vector<std::pair<key_type, int>>
{
    bracketing_item();

    operator bool() const noexcept;

    void set() noexcept;

private:
    bool b_set;
};



template<coefficient_type, vector_type>
struct vector_selector;

template <>
struct vector_selector<coefficient_type::sp_real, vector_type::sparse>
{
    using free_tensor = sparse_tensor<float>;
    using lie = sparse_lie<float>;
};

template <>
struct vector_selector<coefficient_type::sp_real, vector_type::dense>
{
    using free_tensor = dense_tensor<float>;
    using lie = dense_lie<float>;
};

template <>
struct vector_selector<coefficient_type::dp_real, vector_type::sparse>
{
    using free_tensor = sparse_tensor<double>;
    using lie = sparse_lie<double>;
};

template <>
struct vector_selector<coefficient_type::dp_real, vector_type::dense>
{
    using free_tensor = dense_tensor<double>;
    using lie = dense_lie<double>;
};

} // namespace dtl

template <coefficient_type CType, vector_type VType>
using ftensor_type = typename dtl::vector_selector<CType, VType>::free_tensor;

template <coefficient_type CType, vector_type VType>
using lie_type = typename dtl::vector_selector<CType, VType>::lie;



class fallback_context
    : public context
{
    deg_t m_width;
    deg_t m_depth;
    coefficient_type m_ctype;
    std::shared_ptr<tensor_basis> m_tensor_basis;
    std::shared_ptr<lie_basis> m_lie_basis;

    using bracket_result_t = dtl::bracketing_item;
    mutable std::recursive_mutex m_bracketing_lock;
    mutable std::unordered_map<key_type, bracket_result_t> m_bracketing_cache;

    using expand_result_t = std::vector<std::pair<key_type, int>>;
    using expand_let_func = expand_result_t (*)(let_t);
    hall_extension<expand_let_func, dtl::expand_binop> m_expand;

    std::shared_ptr<data_allocator> m_coeff_alloc;
    std::shared_ptr<data_allocator> m_pair_alloc;

public:

    fallback_context(deg_t width, deg_t depth, coefficient_type ctype);

    const std::vector<std::pair<key_type, int>>&
    bracketing(const key_type &key) const;
    const expand_result_t &expand(const key_type &key) const;

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



}// namespace algebra
}// namespace esig


#endif//ESIG_PATHS_SRC_CONTEXT_CONTEXT_CPP_FALLBACK_CONTEXT1_H_
