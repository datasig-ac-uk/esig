//
// Created by user on 10/02/23.
//

#ifndef ESIG_SRC_PYESIG_PY_CONTEXT_H_
#define ESIG_SRC_PYESIG_PY_CONTEXT_H_

#include <pybind11/pybind11.h>

namespace esig { namespace python {

void init_context(pybind11::module_& m);

}}
#endif//ESIG_SRC_PYESIG_PY_CONTEXT_H_
