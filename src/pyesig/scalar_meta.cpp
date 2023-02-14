//
// Created by sam on 03/01/23.
//

#include "scalar_meta.h"

#include <pybind11/pytypes.h>

#include <esig/scalars.h>

using namespace esig;
using namespace esig::python;

namespace py = pybind11;
using namespace pybind11::literals;
extern "C" {

static PyMethodDef PyScalarMetaType_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

PyObject *PyScalarMetaType_call(PyObject *, PyObject *, PyObject *) {
    PyErr_SetString(PyExc_AssertionError, "doh");
    return nullptr;
}

static PyTypeObject PyScalarMetaType_type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    .tp_name = "esig.ScalarMeta",
    .tp_basicsize = sizeof(PyScalarMetaType),
    .tp_itemsize = 0,
    .tp_call = PyScalarMetaType_call,
    .tp_flags = Py_TPFLAGS_TYPE_SUBCLASS,
    .tp_doc = PyDoc_STR("Scalar meta class"),
    .tp_methods = PyScalarMetaType_methods
};

}

void PyScalarMetaType_dealloc(PyObject * arg) {
    PyTypeObject* tp = Py_TYPE(arg);
    PyMem_Free(reinterpret_cast<PyScalarMetaType*>(arg)->ht_name);

    tp->tp_free(arg);
    Py_DECREF(tp);
}



pybind11::handle esig::python::get_scalar_metaclass() {
    assert(PyType_Ready(&PyScalarMetaType_type) == 0);
    return py::handle(reinterpret_cast<PyObject*>(&PyScalarMetaType_type));
}


static PyMethodDef PyScalarTypeBase_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject PyScalarTypeBase_type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "esig.ScalarTypeBase",
    .tp_basicsize = sizeof(PyScalarTypeBase),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    .tp_doc = PyDoc_STR("Base class for scalar type"),
    .tp_methods = PyScalarTypeBase_methods
};

pybind11::handle esig::python::get_scalar_baseclass() {
    assert(PyType_Ready(&PyScalarTypeBase_type) == 0);
    return pybind11::handle(reinterpret_cast<PyObject*>(&PyScalarTypeBase_type));
}

static std::unordered_map<const scalars::scalar_type*, py::object> ctype_type_cache;
//static std::mutex ctype_type_lock;

void esig::python::register_scalar_type(const scalars::scalar_type *ctype, pybind11::handle py_type) {
    auto& found = ctype_type_cache[ctype];
    if (static_cast<bool>(found)) {
        throw std::runtime_error("ctype already registered");
    }

    found = py::reinterpret_borrow<py::object>(py_type);
}

pybind11::object esig::python::to_ctype_type(const scalars::scalar_type *type) {
    // The GIL must be held because we're working with Python objects anyway.
    const auto found = ctype_type_cache.find(type);
    if (found != ctype_type_cache.end()) {
        return found->second;
    }
    throw std::runtime_error("no matching ctype for type " + type->info().name);
}

void esig::python::init_scalar_metaclass(pybind11::module_ &m) {

    PyScalarMetaType_type.tp_base = &PyType_Type;
    if (PyType_Ready(&PyScalarMetaType_type) < 0) {
        throw py::error_already_set();
    }

    m.add_object("ScalarMeta", reinterpret_cast<PyObject*>(&PyScalarMetaType_type));

    Py_INCREF(&PyScalarMetaType_type);
    Py_SET_TYPE(&PyScalarTypeBase_type, &PyScalarMetaType_type);
    if (PyType_Ready(&PyScalarTypeBase_type) < 0) {
        pybind11::pybind11_fail(pybind11::detail::error_string());

    }

    m.add_object("ScalarTypeBase", get_scalar_baseclass());



}
