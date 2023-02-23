//
// Created by user on 20/02/23.
//

#ifndef ESIG_SRC_PYESIG_PY_TO_BUFFER_H_
#define ESIG_SRC_PYESIG_PY_TO_BUFFER_H_

#include "py_esig.h"

#include <vector>

#include <esig/scalars.h>

namespace esig { namespace python {


struct py_to_buffer_options {
    /// Scalar type to use. If null, will be set to the resulting type
    const scalars::scalar_type *type = nullptr;

    /// Maximum number of nested objects to search. Set to 0 for no recursion.
    dimn_t max_nested = 0;

    /// Information about the constructed array
    std::vector<idimn_t> shape;

    /// Allow a single, untagged scalar as argument
    bool allow_scalar = true;

    /// Do not check std library types or imported data types.
    /// All Python types will (try) to be converted to double.
    bool no_check_imported = false;
};


std::string py_buffer_to_type_id(const py::buffer_info& info);
const scalars::scalar_type* py_buffer_fmt_to_stype(const std::string& fmt);

const scalars::scalar_type* scalar_type_for_pytype(py::handle object);

void assign_py_object_to_scalar(scalars::scalar_pointer p, py::handle object);





scalars::key_scalar_array py_to_buffer(py::handle arg, py_to_buffer_options& options);



} // namespace python
} // namespace esig



#endif//ESIG_SRC_PYESIG_PY_TO_BUFFER_H_
