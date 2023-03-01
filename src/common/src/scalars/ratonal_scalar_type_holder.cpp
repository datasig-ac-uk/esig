//
// Created by user on 22/11/22.
//


#include "rational_type.h"


using namespace esig;
using namespace scalars;

template <>
const ScalarType * dtl::scalar_type_holder<rational_scalar_type>::get_type() noexcept
{
    static const rational_type rtype;
    return &rtype;
}
