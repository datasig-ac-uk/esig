//
// Created by user on 06/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_NON_TEMPLATED_IMPL_HALL_BASIS_MAP_H_
#define ESIG_PATHS_SRC_ALGEBRA_NON_TEMPLATED_IMPL_HALL_BASIS_MAP_H_
#include <esig/implementation_types.h>
#include "hall_set.h"

#include <map>
#include <memory>
#include <mutex>

namespace esig {
namespace algebra {
namespace dtl {


class hall_basis_map
{
    std::map<unsigned, std::shared_ptr<hall_set>> m_data;
    std::mutex m_access;
public:
    hall_basis_map() noexcept;

    std::shared_ptr<hall_set> get(deg_t width, deg_t depth);
};

} // namespace dtl
} // namespace algebra_old
} // namespace esig_paths


#endif//ESIG_PATHS_SRC_ALGEBRA_NON_TEMPLATED_IMPL_HALL_BASIS_MAP_H_
