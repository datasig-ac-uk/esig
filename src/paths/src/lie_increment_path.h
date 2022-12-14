//
// Created by user on 31/03/2022.
//

#ifndef ESIG_PATHS_SRC_PATHS_SRC_LIE_INCREMENT_PATH_H_
#define ESIG_PATHS_SRC_PATHS_SRC_LIE_INCREMENT_PATH_H_

#include <esig/implementation_types.h>
#include <esig/paths/path.h>

#include <boost/container/flat_map.hpp>

namespace esig {
namespace paths {


class lie_increment_path : public dyadic_caching_layer
{
    scalars::owned_scalar_array m_buffer;
    boost::container::flat_map<param_t, std::pair<const char*, const char*>> m_data;

public:
    lie_increment_path(
            scalars::owned_scalar_array&& buffer,
            const std::vector<param_t>& indices,
            path_metadata metadata
            );


    using dyadic_caching_layer::log_signature;
    bool empty(const interval &domain) const override;
    algebra::lie log_signature(const interval &domain, const algebra::context &ctx) const override;
};

}// namespace paths
}// namespace esig

#endif//ESIG_PATHS_SRC_PATHS_SRC_LIE_INCREMENT_PATH_H_
