//
// Created by user on 21/03/2022.
//

#include <esig/algebra/context.h>

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include "lite_context.h"

namespace esig {
namespace algebra {

void context::lie_to_tensor_fallback(free_tensor &result, const lie &arg) const {



}
void context::tensor_to_lie_fallback(lie &result, const free_tensor &arg) const {
}


void context::cbh_fallback(free_tensor &collector, const std::vector<lie> &lies) const {
    for (const auto& alie : lies)  {
        collector.fmexp(this->lie_to_tensor(alie));
    }
}
free_tensor context::to_signature(const lie &log_sig) const {
    return this->lie_to_tensor(log_sig).exp();
}
bool context::check_compatible(const context &other) const noexcept {
    return other.width() == width();
}

free_tensor context::zero_tensor(VectorType vtype) const
{
    return construct_tensor({ .vect_type=vtype });
}
lie context::zero_lie(VectorType vtype) const
{
    return construct_lie({ scalars::KeyScalarArray(), vtype });
}
int context_maker::get_priority(const std::vector<std::string> &preferences) const noexcept
{
    return 0;
}


static std::mutex s_context_lock;


static std::vector<std::unique_ptr<context_maker>>& get_context_maker_list() noexcept
{
    static std::vector<std::unique_ptr<context_maker>> list = []() {
        std::vector<std::unique_ptr<context_maker>> tmp;
        tmp.reserve(1);
        tmp.emplace_back(new esig::algebra::lite_context_maker());
        return tmp;
    }();
    return list;
}

std::shared_ptr<const context> get_context(deg_t width, deg_t depth, const scalars::ScalarType * ctype, const std::vector<std::string>& preferences)
{
    std::lock_guard<std::mutex> access(s_context_lock);
    std::vector<std::pair<int, const context_maker*>> found;
    for (const auto& maker : get_context_maker_list()) {
        if (maker->can_get(width, depth, ctype)) {
            found.emplace_back(maker->get_priority(preferences), maker.get());
        }
    }
    auto chosen = std::max_element(found.begin(), found.end(),
                          [](const std::pair<int, const context_maker*>& p1,
                             const std::pair<int, const context_maker*>& p2
                             )
                          {
                              return p1.first < p2.first;
                          });
    return chosen->second->get_context(width, depth, ctype);
}
const context_maker* register_context_maker(std::unique_ptr<context_maker> maker)
{
    std::lock_guard<std::mutex> access(s_context_lock);
    auto& lst = get_context_maker_list();
    lst.push_back(std::move(maker));
    return lst.back().get();
}


} // namespace algebra
} // namespace esig
