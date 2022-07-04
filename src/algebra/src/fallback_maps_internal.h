//
// Created by sam on 04/07/22.
//

#ifndef ESIG_SRC_ALGEBRA_SRC_FALLBACK_MAPS_INTERNAL_H_
#define ESIG_SRC_ALGEBRA_SRC_FALLBACK_MAPS_INTERNAL_H_

#include "fallback_context.h"
#include <esig/implementation_types.h>


namespace esig {
namespace algebra {


template<coefficient_type CType, vector_type VType>
lie_type<CType, VType> tensor_to_lie_impl(const fallback_context &ctx, const ftensor_type<CType, VType> &arg) {
    lie_type<CType, VType> result(ctx.get_lie_basis());
    auto iter = arg.iterate_kv();
    while (iter.advance()) {
        assert(iter.key() < ctx.tensor_size(ctx.depth()));
        for (auto p : ctx.bracketing(iter.key())) {
            result[p.first] += iter.value() * static_cast<typename ftensor_type<CType, VType>::scalar_type>(p.second);
        }
    }
    return result;
}

template<coefficient_type CType, vector_type VType>
ftensor_type<CType, VType> lie_to_tensor_impl(const fallback_context &ctx, const lie_type<CType, VType> &arg) {
    ftensor_type<CType, VType> result(ctx.get_tensor_basis());
    auto iter = arg.iterate_kv();
    while (iter.advance()) {
        for (auto p : ctx.expand(iter.key())) {
            result[p.first] += iter.value() * static_cast<typename ftensor_type<CType, VType>::scalar_type>(p.second);
        }
    }

    return result;
}

template<coefficient_type CType, vector_type VType>
lie_type<CType, VType> cbh_impl(const fallback_context &ctx, const std::vector<lie> &lies) {
    deg_t max_deg = 0;
    std::vector<const lie_type<CType, VType> *> cast_lies;
    cast_lies.reserve(lies.size());
    for (const auto &x : lies) {
        max_deg = std::max(max_deg, x.degree());
        cast_lies.push_back(&lie_base_access::get<lie_type<CType, VType>>(x));
    }

    ftensor_type<CType, VType> result(ctx.get_tensor_basis());
    for (auto x : cast_lies) {
        result.fmexp_inplace(lie_to_tensor_impl<CType, VType>(ctx, *x));
    }

    return tensor_to_lie_impl<CType, VType>(ctx, log(result));
}

} // namespace algebra
} // namespace esig

#endif//ESIG_SRC_ALGEBRA_SRC_FALLBACK_MAPS_INTERNAL_H_
