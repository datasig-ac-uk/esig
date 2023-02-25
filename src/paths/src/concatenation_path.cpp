//
// Created by sam on 19/08/22.
//

#include "esig/paths/concatenation_path.h"

namespace esig {
namespace paths {
algebra::lie concatenation_path::log_signature(const interval &domain, const algebra::context &ctx) const {

    std::vector<algebra::lie> lies;
    lies.reserve(m_paths.size());

    for (const auto& p : m_paths) {
        if (p->empty(domain)) {
            lies.push_back(p->log_signature(domain, ctx));
        }
    }

    return ctx.cbh(lies, metadata().result_vec_type);
}
bool concatenation_path::empty(const interval &domain) const {
    for (const auto& p : m_paths) {
        if (!p->empty(domain)) {
            return false;
        }
    }
    return true;
}
}// namespace paths
}// namespace esig
