//
// Created by user on 09/12/22.
//

#include "scalar_type.h"

#include "scalar_meta.h"

using namespace esig::scalars;


void esig::python::init_scalar_type(py::module_ &m) {

    init_scalar_metaclass(m);

    make_scalar_type(m, scalars::dtl::scalar_type_holder<float>::get_type());
    make_scalar_type(m, scalars::dtl::scalar_type_holder<double>::get_type());
    make_scalar_type(m, scalars::dtl::scalar_type_holder<rational_scalar_type>::get_type());

}
