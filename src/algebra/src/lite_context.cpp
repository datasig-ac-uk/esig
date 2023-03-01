//
// Created by sam on 04/01/23.
//

#include "lite_context.h"

#include <boost/functional/hash.hpp>

namespace esig {
namespace algebra {
bool lite_context_maker::can_get(deg_t deg, deg_t deg_1, const scalars::ScalarType *type) const noexcept {
    return true;
}
int lite_context_maker::get_priority(const std::vector<std::string> &preferences) const noexcept {
    return 0;
}

static std::unordered_map<std::tuple<deg_t, deg_t, const scalars::ScalarType *>, std::shared_ptr<const context>,
    boost::hash<std::tuple<deg_t, deg_t, const scalars::ScalarType *>>> s_lite_context_cache;


std::shared_ptr<const context> lite_context_maker::get_context(deg_t width, deg_t depth, const scalars::ScalarType *type) const {
    // We should already behind a mutex lock
    auto& found = s_lite_context_cache[std::make_tuple(width, depth, type)];
    if (!static_cast<bool>(found)) {
        if (type == scalars::dtl::scalar_type_holder<float>::get_type()) {
            found = std::shared_ptr<const context>(new lite_context<lal::float_field>(width, depth));
        } else if (type == scalars::dtl::scalar_type_holder<double>::get_type()) {
            found = std::shared_ptr<const context>(new lite_context<lal::double_field>(width, depth));
        } else {
            throw std::runtime_error("bad scalar_type");
        }
    }


    return found;
}



}// namespace algebra
}// namespace esig
