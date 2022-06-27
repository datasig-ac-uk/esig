//
// Created by user on 27/06/22.
//

#include <esig/algebra/python_interface.h>

pybind11::dtype esig::algebra::dtype_from(esig::algebra::coefficient_type ctype)
{
#define ESIG_SWITCH_FN(ct) pybind11::dtype::of<esig::algebra::type_of_coeff<ct>>()
    ESIG_MAKE_CTYPE_SWITCH(ctype)
#undef ESIG_SWITCH_FN
}
