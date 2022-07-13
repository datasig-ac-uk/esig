//
// Created by user on 05/07/22.
//

#ifndef ESIG_SRC_ALGEBRA_SRC_FALLBACK_MAPS_H_
#define ESIG_SRC_ALGEBRA_SRC_FALLBACK_MAPS_H_

#include <esig/implementation_types.h>
#include "lie_basis/lie_basis.h"
#include "lie_basis/hall_extension.h"
#include "tensor_basis/tensor_basis.h"
#include "sparse_lie.h"
#include "dense_lie.h"
#include "sparse_tensor.h"
#include "dense_tensor.h"

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace esig {
namespace algebra {

namespace dtl {
class expand_binop {
    const tensor_basis* m_basis;

public:
    using result_t = std::vector<std::pair<key_type, int>>;

    explicit expand_binop(const tensor_basis* arg);
    result_t operator()(const result_t &a, const result_t &b) const;
};

struct bracketing_item : std::vector<std::pair<key_type, int>> {
    bracketing_item() = default;

    inline operator bool() const noexcept
    {
        return b_set;
    }

    inline void set() noexcept { b_set = true; }

private:
    bool b_set = false;
};

} // namespace dtl


class maps
{

    std::shared_ptr<tensor_basis> p_tensor_basis;
    std::shared_ptr<lie_basis> p_lie_basis;

    static std::vector<std::pair<key_type, int>> expand_func(let_t letter);

public:
    hall_extension<decltype(&expand_func), dtl::expand_binop> expand;

private:
    mutable std::recursive_mutex m_bracketing_lock;
    mutable std::unordered_map<key_type, dtl::bracketing_item> m_bracketing_cache;

public:
    const std::vector<std::pair<key_type, int>>& rbracket(key_type key) const;


    maps(std::shared_ptr<tensor_basis> tbasis, std::shared_ptr<lie_basis> lbasis);

    template <typename Lie, typename Tensor>
    Tensor lie_to_tensor(const Lie& arg) const;

    template <typename Lie, typename Tensor>
    Lie tensor_to_lie(const Tensor& arg) const;

    template <typename InputIt>
    typename std::iterator_traits<InputIt>::value_type
    cbh(InputIt begin, InputIt end) const;


};



template<typename Lie, typename Tensor>
Tensor maps::lie_to_tensor(const Lie &arg) const {
    using scal_t = typename Lie::scalar_type;
    Tensor result(p_tensor_basis);
    auto iter = arg.iterate_kv();
    while (iter.advance()) {
        for (auto p : expand(iter.key())) {
            result.add_scal_prod(p.first, iter.value()*scal_t(p.second));
        }
    }
    return result;
}
template<typename Lie, typename Tensor>
Lie maps::tensor_to_lie(const Tensor &arg) const {
    using scal_t = typename Lie::scalar_type;
    Lie result(p_lie_basis);
    auto iter = arg.iterate_kv();
    while (iter.advance()) {
        if (iter.value() != scal_t(0)) {
            for (auto p : rbracket(iter.key())) {
                auto deg = p_lie_basis->degree(p.first);
                result.add_scal_prod(p.first, iter.value()*scal_t(p.second)/scal_t(deg));
            }
        }
    }
    return result;
}

namespace dtl {

template <typename Lie>
struct matching_tensor_selector;

template <typename Lie>
using matching_tensor = typename matching_tensor_selector<Lie>::type;

template <typename Tensor>
struct matching_lie_selector;

template <typename Tensor>
using matching_lie = typename matching_lie_selector<Tensor>::type;


template <typename S>
struct matching_tensor_selector<sparse_lie<S>>
{
    using type = sparse_tensor<S>;
};
template <typename S>
struct matching_tensor_selector<dense_lie<S>>
{
    using type = dense_tensor<S>;
};
template <typename S>
struct matching_lie_selector<sparse_tensor<S>>
{
    using type = sparse_lie<S>;
};
template<typename S>
struct matching_lie_selector<dense_tensor<S>>
{
    using type = dense_lie<S>;
};


}


template<typename InputIt>
typename std::iterator_traits<InputIt>::value_type
maps::cbh(InputIt begin, InputIt end) const {
    using lie_t = typename std::iterator_traits<InputIt>::value_type;
    using scal_t = typename lie_t::scalar_type;
    using tensor_t = dtl::matching_tensor<lie_t>;

    tensor_t tmp(p_tensor_basis, scal_t(1));
    for (auto it = begin; it != end; ++it) {
        tmp.fmexp_inplace(lie_to_tensor<lie_t, tensor_t>(*it));
    }
    return tensor_to_lie<lie_t, tensor_t>(log(tmp));
}

} // namespace algebra
} // namespace esig
#endif//ESIG_SRC_ALGEBRA_SRC_FALLBACK_MAPS_H_
