//
// Created by sam on 19/08/22.
//

#ifndef ESIG_SRC_PATHS_SRC_PY_PATHS_KWARGS_TO_CONTEXT_H_
#define ESIG_SRC_PATHS_SRC_PY_PATHS_KWARGS_TO_CONTEXT_H_

#include <esig/paths/path.h>
#include <esig/algebra/context.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace esig {
namespace paths {

std::shared_ptr<const algebra::context> kwargs_to_context(const path_metadata& md, const pybind11::kwargs &kwargs);

} // namespace paths
} // namespace esig


#endif//ESIG_SRC_PATHS_SRC_PY_PATHS_KWARGS_TO_CONTEXT_H_
