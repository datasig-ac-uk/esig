//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_GET_VECTOR_CONSTRUCTION_DATA_H_
#define ESIG_SRC_PYESIG_GET_VECTOR_CONSTRUCTION_DATA_H_

#include "py_esig.h"

#include <esig/algebra/context.h>

namespace esig { namespace python {

algebra::vector_construction_data get_vector_construction_data(const py::object& data, const py::kwargs& kwargs);

} // namespace python
} // namespace esig

#endif//ESIG_SRC_PYESIG_GET_VECTOR_CONSTRUCTION_DATA_H_
