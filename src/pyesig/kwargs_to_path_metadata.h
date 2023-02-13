//
// Created by user on 11/02/23.
//

#ifndef ESIG_SRC_PYESIG_KWARGS_TO_PATH_METADATA_H_
#define ESIG_SRC_PYESIG_KWARGS_TO_PATH_METADATA_H_

#include "py_esig.h"

#include <esig/paths/path.h>

namespace esig {
namespace python {

paths::path_metadata kwargs_to_metadata(const py::kwargs& kwargs);


} // namespace python
} // namespace esig

#endif//ESIG_SRC_PYESIG_KWARGS_TO_PATH_METADATA_H_
