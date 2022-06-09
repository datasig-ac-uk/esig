//
// Created by sam on 02/05/22.
//

#ifndef ESIG_PY_LIE_INCREMENT_PATH_H
#define ESIG_PY_LIE_INCREMENT_PATH_H

#include "../lie_increment_path.h"
#include "esig/paths/path.h"

#include <pybind11/pybind11.h>

namespace esig {
namespace paths {


void init_lie_increment_path(pybind11::module_& m);

path construct_lie_increment_path(const pybind11::args &args, const pybind11::kwargs& kwargs);


} // namespace paths
} // namespace esig


#endif//ESIG_PY_LIE_INCREMENT_PATH_H
