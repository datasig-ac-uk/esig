//
// Created by user on 08/12/22.
//

#ifndef ESIG_SRC_PYESIG_PY_SCALARS_H_
#define ESIG_SRC_PYESIG_PY_SCALARS_H_

#include "py_esig.h"

#include <functional>

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

    /// cleanup function to be called when we're finished with the data
    std::function<void(void)> cleanup = nullptr;
};

char format_to_type_char(const std::string& fmt);
std::string py_buffer_to_type_id(const py::buffer_info& info);


const scalars::scalar_type* py_buffer_to_scalar_type(const py::buffer_info& info);
const scalars::scalar_type* py_type_to_scalar_type(const py::type& type);
const scalars::scalar_type* py_arg_to_ctype(const py::object& object);

py::type scalar_type_to_py_type(const scalars::scalar_type*);
void assign_py_object_to_scalar(scalars::scalar_pointer ptr, py::handle object);

scalars::key_scalar_array
py_to_buffer(const py::object& object, py_to_buffer_options& options);


void init_scalars(py::module_& m);


} // namespace python
} // namespace esig

#endif//ESIG_SRC_PYESIG_PY_SCALARS_H_
