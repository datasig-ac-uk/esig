//
// Created by sam on 03/01/23.
//

#include "scalar_meta.h"

#include <esig/scalars.h>


using namespace esig;
using namespace esig::python;

namespace py = pybind11;

struct scalar_meta {
    PyObject_HEAD
    const scalars::scalar_type* tp_stype;
};


static scalars::scalar ScalarMeta_call(const scalar_meta& meta, const py::args& args, const py::kwargs& kwargs) {



    return scalars::scalar(meta.tp_stype);
}



py::handle init_scalar_metaclass(pybind11::module_ &m) {

    namespace py = pybind11;
    using namespace pybind11::literals;


    py::class_<scalar_meta> cls(m, "ScalarMeta",
                                py::custom_type_setup([](PyHeapTypeObject* heap_type) {
                                    auto* type = &heap_type->ht_type;

                                    Py_INCREF(&PyType_Type);
                                    type->tp_base = &PyType_Type;
                                }));
    cls.def("__call__", &ScalarMeta_call);


    return cls;
}
