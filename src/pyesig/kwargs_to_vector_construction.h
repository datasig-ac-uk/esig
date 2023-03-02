//
// Created by user on 09/12/22.
//

#ifndef ESIG_SRC_PYESIG_KWARGS_TO_VECTOR_CONSTRUCTION_H_
#define ESIG_SRC_PYESIG_KWARGS_TO_VECTOR_CONSTRUCTION_H_

#include "py_esig.h"

#include <esig/scalars.h>
#include <esig/algebra/context_fwd.h>
#include <esig/algebra/algebra_fwd.h>

namespace esig { namespace python {

struct py_vector_construction_helper {
    /// Context if provided by user
    std::shared_ptr<const algebra::context> ctx = nullptr;
    /// Width and depth
    deg_t width = 0;
    deg_t depth = 0;
    /// Coefficient type
    const scalars::ScalarType *ctype = nullptr;
    /// Vector type to be requested
    algebra::VectorType vtype = algebra::VectorType::dense;
    /// flags for saying if the user explicitly requested ctype and vtype
    bool ctype_requested = false;
    bool vtype_requested = false;
    /// Data type provided
};


py_vector_construction_helper kwargs_to_construction_data(const py::kwargs& kwargs);

}}


#endif//ESIG_SRC_PYESIG_KWARGS_TO_VECTOR_CONSTRUCTION_H_
