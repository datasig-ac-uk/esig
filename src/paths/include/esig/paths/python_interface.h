//
// Created by sam on 03/05/22.
//

#ifndef ESIG_PATHS_PYTHON_INTERFACE_H_
#define ESIG_PATHS_PYTHON_INTERFACE_H_

#include <pybind11/pybind11.h>

#include <esig/paths/path.h>

#include <functional>
#include <utility>

namespace esig {
namespace paths {

using path_constructor = std::function<path(const pybind11::args &, const pybind11::kwargs &)>;


void register_pypath_constructor(const pybind11::type &tp, path_constructor &&ctor);


}// namespace paths
}// namespace esig


#endif//ESIG_PATHS_PYTHON_INTERFACE_H_