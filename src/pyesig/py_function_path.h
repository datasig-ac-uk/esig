//
// Created by user on 20/02/23.
//

#ifndef ESIG_SRC_PYESIG_PY_FUNCTION_PATH_H_
#define ESIG_SRC_PYESIG_PY_FUNCTION_PATH_H_

#include "py_esig.h"

#include <esig/paths/path.h>

namespace esig {
namespace python {

class py_function_path : public paths::dynamically_constructed_path
{
    py::function m_func;

public:
    py_function_path(py::function func, paths::path_metadata md)
        : paths::dynamically_constructed_path(md), m_func(func)
    {}


};

}// namespace python
}// namespace esig

#endif//ESIG_SRC_PYESIG_PY_FUNCTION_PATH_H_
