//
// Created by user on 20/03/2022.
//

#include "fallback_context.h"
#include "lie_basis/lie_basis.h"
#include "tensor_basis/tensor_basis.h"
#include <esig/implementation_types.h>
#include <mutex>

#include <boost/functional/hash.hpp>

namespace esig {
namespace algebra {

//std::shared_ptr<tensor_basis> esig::algebra::fallback_context::tbasis() const noexcept {
//    return m_tensor_basis;
//}
//std::shared_ptr<lie_basis> esig::algebra::fallback_context::lbasis() const noexcept {
//    return m_lie_basis;
//}

static register_maker_helper<fallback_context_maker> fb_context_maker;

class fallback_context_cache {
    using key_type = std::tuple<deg_t, deg_t, const scalars::scalar_type *>;
    using hash = boost::hash<key_type>;

    std::mutex m_lock;
    std::unordered_map<key_type, std::shared_ptr<const context>, hash> m_cache;

    std::shared_ptr<const context> make(deg_t width, deg_t depth, const scalars::scalar_type *ctype) {
        if (ctype == ::esig::scalars::dtl::scalar_type_holder<float>::get_type()) {
            return std::shared_ptr<const context>(new fallback_context<float>(width, depth));
        } else if (ctype == ::esig::scalars::dtl::scalar_type_holder<double>::get_type()) {
            return std::shared_ptr<const context>(new fallback_context<double>(width, depth));
        }

        throw std::runtime_error("no context for this coefficient type");

        return {nullptr};
    }

public:
    std::shared_ptr<const context> get(deg_t width, deg_t depth, const scalars::scalar_type *ctype) {
        std::lock_guard<std::mutex> access(m_lock);

        std::tuple<deg_t, deg_t, const scalars::scalar_type *> key(width, depth, ctype);
        auto &result = m_cache[key];
        if (result) {
            return result;
        }
        result = make(width, depth, ctype);
        return result;
    }
};

static fallback_context_cache s_cache;

std::shared_ptr<const context> get_fallback_context(deg_t width, deg_t depth, const scalars::scalar_type *ctype) {
    return s_cache.get(width, depth, ctype);
}

std::shared_ptr<const context> fallback_context_maker::get_context(deg_t width, deg_t depth, const scalars::scalar_type *ctype) const {
    return s_cache.get(width, depth, ctype);
}
bool fallback_context_maker::can_get(deg_t width, deg_t depth, const scalars::scalar_type *ctype) const noexcept {
    return true;
}
int fallback_context_maker::get_priority(const std::vector<std::string> &preferences) const noexcept {
    return 0;
}
//
//template<typename VectorInner, typename VectorWrapper>
//VectorWrapper convert_impl(const VectorWrapper &arg)
//{
//    using scalar_type = typename VectorInner::scalar_type;
//
//    const auto& base = algebra_base_access::get<(arg);
//    if (typeid(base) == typeid(VectorInner)) {
//        return arg;
//    }
//
//    VectorInner result;
//
//    for (auto it : base) {
//        result.add_scal_prod(it.key(), scalar_type(it.value()))
//    }
//
//    return VectorWrapper(std::move(result));
//}
//
//free_tensor fallback_context::convert(const free_tensor &arg) const {
//#define ESIG_SWITCH_FN(CTYPE, VTYPE) convert_impl<ftensor_type<(CTYPE), (VTYPE)>>(arg)
//    ESIG_MAKE_SWITCH(m_ctype, arg.storage_type())
//#undef ESIG_SWITCH_FN
//}
//lie fallback_context::convert(const lie &arg) const {
//#define ESIG_SWITCH_FN(CTYPE, VTYPE) convert_impl<lie_type<(CTYPE), (VTYPE)>>(arg)
//    ESIG_MAKE_SWITCH(m_ctype, arg.storage_type())
//#undef ESIG_SWITCH_FN
//}

}// namespace algebra
}// namespace esig
