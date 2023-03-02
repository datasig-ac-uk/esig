//
// Created by sam on 19/08/22.
//

#ifndef ESIG_SRC_PATHS_SRC_CONCATENATION_PATH_H_
#define ESIG_SRC_PATHS_SRC_CONCATENATION_PATH_H_

#include "path.h"

namespace esig {
namespace paths {


class concatenation_path : public path_interface
{
    std::vector<std::unique_ptr<const path_interface>> m_paths;

public:
    algebra::Lie log_signature(const interval &domain, const algebra::context &ctx) const override;

    bool empty(const interval &domain) const override;
};

}// namespace paths
}// namespace esig

#endif//ESIG_SRC_PATHS_SRC_CONCATENATION_PATH_H_
