//
// Created by user on 05/04/2022.
//

#ifndef ESIG_PATHS_SRC_LIBALGEBRA_CONTEXT_SRC_LIBALGEBRA_CONTEXT_MAKER_H_
#define ESIG_PATHS_SRC_LIBALGEBRA_CONTEXT_SRC_LIBALGEBRA_CONTEXT_MAKER_H_


#include <esig/libalgebra_context/libalgebra_context.h>
#include <memory>
#include <mutex>
#include <limits>

#include <boost/functional/hash.hpp>

namespace esig {
namespace algebra {

namespace dtl {



}

class libalgebra_context_maker : public context_maker
{
    using config = std::tuple<deg_t, deg_t, coefficient_type>;
    using context_map = dtl::context_map;
    mutable std::unordered_map<config, std::shared_ptr<const context>, boost::hash<config>> cache;
    mutable std::recursive_mutex m_lock;

public:

    const context_map& get() const;


    libalgebra_context_maker();
    bool can_get(deg_t, deg_t, coefficient_type) const noexcept override;
    int get_priority(const std::vector<std::string> &preferences) const noexcept override;
    std::shared_ptr<const context> get_context(deg_t, deg_t, coefficient_type) const override;
};


} // namespace algebra
} // namespace esig

#endif//ESIG_PATHS_SRC_LIBALGEBRA_CONTEXT_SRC_LIBALGEBRA_CONTEXT_MAKER_H_
