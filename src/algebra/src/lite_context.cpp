//
// Created by sam on 04/01/23.
//

#include "lite_context.h"

namespace esig {
namespace algebra {
bool lite_context_maker::can_get(deg_t deg, deg_t deg_1, const scalars::scalar_type *type) const noexcept {
    return true;
}
int lite_context_maker::get_priority(const std::vector<std::string> &preferences) const noexcept {
    return 0;
}
std::shared_ptr<const context> lite_context_maker::get_context(deg_t width, deg_t depth, const scalars::scalar_type *type) const {

    if (type == scalars::dtl::scalar_type_holder<float>::get_type()) {
        return std::shared_ptr<const context>(new lite_context<lal::float_field>(width, depth));
    }
    else if (type == scalars::dtl::scalar_type_holder<double>::get_type()) {
        return std::shared_ptr<const context>(new lite_context<lal::double_field>(width, depth));
    }

    throw std::runtime_error("bad scalar_type");
}


static register_maker_helper<lite_context_maker> lite_maker_helper;

}// namespace algebra
}// namespace esig