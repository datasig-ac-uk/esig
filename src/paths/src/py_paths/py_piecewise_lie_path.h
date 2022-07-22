//
// Created by user on 22/07/22.
//

#ifndef ESIG_SRC_PATHS_SRC_PY_PATHS_PY_PIECEWISE_LIE_PATH_H_
#define ESIG_SRC_PATHS_SRC_PY_PATHS_PY_PIECEWISE_LIE_PATH_H_

#include "../piecewise_lie_path.h"

#include <pybind11/pybind11.h>

namespace esig {
namespace paths {

void init_piecewise_lie_path(pybind11::module_& m);

path construct_piecewise_lie_path(const pybind11::args& args, const pybind11::kwargs& kwargs);


} // namespace paths
} // namespace esig


#endif//ESIG_SRC_PATHS_SRC_PY_PATHS_PY_PIECEWISE_LIE_PATH_H_
