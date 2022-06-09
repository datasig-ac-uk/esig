//
// Created by user on 21/03/2022.
//

#include <esig/algebra/context.h>

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>


namespace esig {
namespace algebra {





coefficient_type signature_data::ctype() const
{
    return m_coeff_type;
}
vector_type signature_data::vtype() const
{
    return m_vector_type;
}




free_tensor context::zero_tensor(vector_type vtype) const
{
    return construct_tensor({ctype(), vtype});
}
lie context::zero_lie(vector_type vtype) const
{
    return construct_lie({ctype(), vtype});
}
int context_maker::get_priority(const std::vector<std::string> &preferences) const noexcept
{
    return 0;
}


static std::mutex s_context_lock;


static std::vector<std::unique_ptr<context_maker>>& get_context_maker_list() noexcept
{
    static std::vector<std::unique_ptr<context_maker>> list;
    return list;
}

std::shared_ptr<context> get_context(deg_t width, deg_t depth, coefficient_type ctype, const std::vector<std::string>& preferences)
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
