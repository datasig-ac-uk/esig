//
// Created by user on 22/07/22.
//

#ifndef ESIG_SRC_PATHS_SRC_PIECEWISE_LIE_PATH_H_
#define ESIG_SRC_PATHS_SRC_PIECEWISE_LIE_PATH_H_

#include <esig/implementation_types.h>
#include <esig/intervals.h>
#include "path.h"

#include <esig/algebra/lie_interface.h>
#include <esig/algebra/tensor_interface.h>

namespace esig {
namespace paths {

class piecewise_lie_path : public path_interface
{
public:
    using lie_piece = std::pair<real_interval, algebra::Lie>;

private:
    std::vector<lie_piece> m_data;

    static algebra::Lie compute_lie_piece(const lie_piece& arg, const interval& domain);

public:

    piecewise_lie_path(std::vector<lie_piece> data, path_metadata metadata);

    using path_interface::log_signature;
    bool empty(const interval& domain) const override;
    algebra::Lie log_signature(const interval &domain, const algebra::context &ctx) const override;
};

}// namespace paths
}// namespace esig

#endif//ESIG_SRC_PATHS_SRC_PIECEWISE_LIE_PATH_H_
