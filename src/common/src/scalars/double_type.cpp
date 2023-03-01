//
// Created by user on 14/11/22.
//

#include "double_type.h"

using namespace esig;
using namespace scalars;

static void float_to_double(void* dst, const void* src)
{
    ::new (dst) double(*static_cast<const float*>(src));
}

static void rational_to_double(void* dst, const void* src)
{
    ::new (dst) double(*static_cast<const rational_scalar_type*>(src));
}

double_type::double_type() : StandardScalarType<double>("f64", "double")
{
}
