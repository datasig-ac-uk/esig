//
// Created by user on 09/12/22.
//

#include "ctype_to_py_fmt.h"


std::string esig::python::ctype_to_py_fmt(const esig::scalars::scalar_type *type) {
    if (type == esig::scalars::dtl::scalar_type_holder<double>::get_type()) {
        return "d";
    }
    if (type == esig::scalars::dtl::scalar_type_holder<float>::get_type()) {
        return "f";
    }
}
