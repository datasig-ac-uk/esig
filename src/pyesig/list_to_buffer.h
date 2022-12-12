//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_LIST_TO_BUFFER_H_
#define ESIG_SRC_PYESIG_LIST_TO_BUFFER_H_

#include "py_esig.h"

#include <utility>
#include <vector>

#include <esig/scalars.h>

namespace esig { namespace python {

scalars::key_scalar_array
list_to_buffer(const py::list& arg, const scalars::scalar_type* ctype);

}}

#endif//ESIG_SRC_PYESIG_LIST_TO_BUFFER_H_
