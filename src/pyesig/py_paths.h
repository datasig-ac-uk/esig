//
// Created by user on 08/12/22.
//

#ifndef ESIG_SRC_PYESIG_PY_PATHS_H_
#define ESIG_SRC_PYESIG_PY_PATHS_H_

#include "py_esig.h"


namespace esig { namespace python {

void buffer_to_indices(std::vector<param_t>& indices, const py::buffer_info& info);


void init_paths(py::module_& m);

} // namespace python
} // namespace esig


#endif//ESIG_SRC_PYESIG_PY_PATHS_H_
