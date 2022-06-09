//
// Created by user on 06/03/2022.
//

#include "hall_basis_map.h"

namespace esig {
namespace algebra {
namespace dtl {

hall_basis_map::hall_basis_map() noexcept : m_data(), m_access()
{
}
std::shared_ptr<hall_set> hall_basis_map::get(deg_t width, deg_t depth)
{
    std::lock_guard<std::mutex> lock(m_access);
    auto& found = m_data[width];
    if (found) {
        found->grow_up(depth);
        return found;
    }

    found.reset(new hall_set(width, depth));
    return found;
}

}
}
}
