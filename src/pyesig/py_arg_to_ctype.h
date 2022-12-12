//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_PY_ARG_TO_CTYPE_H_
#define ESIG_SRC_PYESIG_PY_ARG_TO_CTYPE_H_

#include "py_esig.h"
#include <esig/scalars.h>


namespace esig { namespace python {

const scalars::scalar_type* py_arg_to_ctype(const py::object& arg);

}}

#endif//ESIG_SRC_PYESIG_PY_ARG_TO_CTYPE_H_
