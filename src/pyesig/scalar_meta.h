//
// Created by sam on 03/01/23.
//

#ifndef ESIG_SRC_PYESIG_SCALAR_META_H_
#define ESIG_SRC_PYESIG_SCALAR_META_H_

#include <esig/implementation_types.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <esig/scalars.h>



namespace esig { namespace python {


inline pybind11::handle get_scalar_metaclass() {
    namespace py = pybind11;
    auto esig_mod = py::module_::import("esig._esig");
    return esig_mod.attr("ScalarMeta");
}

inline void make_scalar_type(pybind11::module_& m, const scalars::scalar_type* type) {
    const auto* name = type->info().name.c_str();
    pybind11::capsule holder(type, name);
    m.add_object(name, holder);
}

pybind11::handle init_scalar_metaclass(pybind11::module_& m);

} // namespace python
} // namespace esig


#endif//ESIG_SRC_PYESIG_SCALAR_META_H_
