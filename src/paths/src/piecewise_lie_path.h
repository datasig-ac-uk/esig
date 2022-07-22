//
// Created by user on 22/07/22.
//

#ifndef ESIG_SRC_PATHS_SRC_PIECEWISE_LIE_PATH_H_
#define ESIG_SRC_PATHS_SRC_PIECEWISE_LIE_PATH_H_

#include <esig/implementation_types.h>
#include <esig/intervals.h>
#include <esig/paths/path.h>

#include <esig/algebra/lie_interface.h>
#include <esig/algebra/free_tensor_interface.h>


namespace esig {
namespace paths {

class piecewise_lie_path : public dyadic_caching_layer
{
    using lie_piece = std::pair<real_interval, algebra::lie>;

    std::vector<lie_piece> m_data;

    static algebra::lie compute_lie_piece(const lie_piece& arg, const interval& domain);

public:

    piecewise_lie_path(std::vector<lie_piece> data, path_metadata metadata);

    using dyadic_caching_layer::log_signature;
    bool empty(const interval& domain) const override;
    algebra::lie log_signature(const interval &domain, const algebra::context &ctx) const override;
};

}// namespace paths
}// namespace esig

#endif//ESIG_SRC_PATHS_SRC_PIECEWISE_LIE_PATH_H_