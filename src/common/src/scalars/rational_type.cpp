//
// Created by user on 22/11/22.
//

#include "rational_type.h"

using namespace esig;
using namespace scalars;


static void float_to_rational(void* dst, const void* src)
{
    ::new (dst) rational_scalar_type(*static_cast<const float*>(src));
}

static void double_to_rational(void* dst, const void* src)
{
    ::new (dst) rational_scalar_type(*static_cast<const double*>(src));
}




rational_type::rational_type()
    : StandardScalarType<rational_scalar_type>("rational", "rational")
{
}
