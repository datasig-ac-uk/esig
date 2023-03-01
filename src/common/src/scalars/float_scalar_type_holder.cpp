//
// Created by user on 14/11/22.
//


#include "float_type.h"

using namespace esig;
using namespace scalars;

template <>
const ScalarType * esig::scalars::dtl::scalar_type_holder<float>::get_type() noexcept
{
    static const float_type ftype;
    return &ftype;
}
