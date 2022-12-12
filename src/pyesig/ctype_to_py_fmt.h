//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_CTYPE_TO_PY_FMT_H_
#define ESIG_SRC_PYESIG_CTYPE_TO_PY_FMT_H_

#include "py_esig.h"
#include <esig/scalars.h>

namespace esig { namespace python {

std::string ctype_to_py_fmt(const scalars::scalar_type* type);

}}

#endif//ESIG_SRC_PYESIG_CTYPE_TO_PY_FMT_H_
