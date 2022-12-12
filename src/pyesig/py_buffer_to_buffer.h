//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_PY_BUFFER_TO_BUFFER_H_
#define ESIG_SRC_PYESIG_PY_BUFFER_TO_BUFFER_H_

#include "py_esig.h"

#include <esig/scalars.h>

namespace esig { namespace python {

scalars::owned_scalar_array
py_buffer_to_buffer(const py::buffer_info& buf, const scalars::scalar_type* type);


}}


#endif//ESIG_SRC_PYESIG_PY_BUFFER_TO_BUFFER_H_
