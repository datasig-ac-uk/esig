//
// Created by sam on 03/01/23.
//

#ifndef ESIG_SRC_PYESIG_SCALAR_META_H_
#define ESIG_SRC_PYESIG_SCALAR_META_H_

#include <esig/implementation_types.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <esig/scalars.h>

extern "C" PYBIND11_EXPORT void PyScalarMetaType_dealloc(PyObject *);

namespace esig { namespace python {

struct PyScalarMetaType {
    PyHeapTypeObject tp_hto;
    char* ht_name;
    const scalars::scalar_type* tp_ctype;
};


struct PyScalarTypeBase {
    PyObject_VAR_HEAD;
};

pybind11::handle PYBIND11_EXPORT get_scalar_metaclass();
pybind11::handle PYBIND11_EXPORT get_scalar_baseclass();
//
//inline void make_scalar_type(pybind11::module_& m, const scalars::scalar_type* type) {
//    const auto* name = type->info().name.c_str();
//    pybind11::capsule holder(type, name);
//    m.add_object(name, holder);
//}


namespace dtl {

struct new_scalar_type_temps_manager {
    char* ht_tpname = nullptr;
    char* ht_name = nullptr;
    char* tp_doc = nullptr;
    PyScalarMetaType* cls = nullptr;

    ~new_scalar_type_temps_manager() {
        if (PyErr_Occurred()) {
            Py_CLEAR(cls);
        }

        PyMem_Free(ht_name);
        PyMem_Free(ht_tpname);
        PyMem_Free(tp_doc);
    }

};

} // namespace dtl

void register_scalar_type(const scalars::scalar_type* ctype, pybind11::handle py_type);


inline void make_scalar_type(pybind11::module_& m, const scalars::scalar_type* ctype) {
    namespace py = pybind11;

    py::object mcs = py::reinterpret_borrow<py::object>(get_scalar_metaclass());

    py::handle base = get_scalar_baseclass();
//    py::handle bases(PyTuple_Pack(1, base.ptr()));
//    py::handle base((PyObject*) &PyBaseObject_Type);


    auto* mcs_tp = reinterpret_cast<PyTypeObject*>(mcs.ptr());

    const auto &name = ctype->info().name;
    py::str ht_name(name);
    dtl::new_scalar_type_temps_manager tmp_manager;

    py::object ht_module(m);

    tmp_manager.ht_name = reinterpret_cast<char*>(PyMem_Malloc(name.size()+1));
    if (tmp_manager.ht_name == nullptr) {
        PyErr_NoMemory();
        throw py::error_already_set();
    }
    memcpy(tmp_manager.ht_name, name.c_str(), name.size());

    tmp_manager.cls = reinterpret_cast<PyScalarMetaType*>(mcs_tp->tp_alloc(mcs_tp, 0));
    if (tmp_manager.cls == nullptr) {
        throw py::error_already_set();
    }
    auto* hto = reinterpret_cast<PyHeapTypeObject*>(&tmp_manager.cls->tp_hto);
    auto* type = &hto->ht_type;

    type->tp_flags = (Py_TPFLAGS_DEFAULT
                      | Py_TPFLAGS_HEAPTYPE
                      | Py_TPFLAGS_DISALLOW_INSTANTIATION
                      | Py_TPFLAGS_HAVE_GC);
    hto->ht_module = ht_module.release().ptr();

    type->tp_as_async = &hto->as_async;
    type->tp_as_buffer = &hto->as_buffer;
    type->tp_as_sequence = &hto->as_sequence;
    type->tp_as_mapping = &hto->as_mapping;
    type->tp_as_number = &hto->as_number;

    type->tp_base = reinterpret_cast<PyTypeObject*>(base.inc_ref().ptr());
//    type->tp_bases = bases.ptr();

    type->tp_doc = tmp_manager.tp_doc;

    hto->ht_qualname = ht_name.release().ptr();
    hto->ht_name = hto->ht_qualname;

    type->tp_name = tmp_manager.ht_name;
    tmp_manager.ht_name = nullptr;

    type->tp_basicsize = sizeof(PyScalarMetaType);
    type->tp_itemsize = 0;

    type->tp_dealloc = PyScalarMetaType_dealloc;

    tmp_manager.cls->tp_ctype = ctype;

    if (PyType_Ready(type) < 0) {
        pybind11::pybind11_fail("Error " + pybind11::detail::error_string());
    }

    py::handle h_class(reinterpret_cast<PyObject *>(tmp_manager.cls));
    register_scalar_type(ctype, h_class);
    m.add_object(name.c_str(), h_class);

}


inline const scalars::scalar_type* to_stype_ptr(const pybind11::handle& arg) {
    if (!pybind11::isinstance(arg, get_scalar_metaclass())) {
        throw pybind11::type_error("argument is not a valid scalar type");
    }
    return reinterpret_cast<PyScalarMetaType*>(arg.ptr())->tp_ctype;
}

PYBIND11_EXPORT pybind11::object to_ctype_type(const scalars::scalar_type* type);


void init_scalar_metaclass(pybind11::module_& m);

} // namespace python
} // namespace esig


#endif//ESIG_SRC_PYESIG_SCALAR_META_H_
