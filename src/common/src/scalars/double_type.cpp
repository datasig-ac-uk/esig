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

double_type::double_type() : standard_scalar_type<double>("f64", "double")
{
    standard_scalar_type<double>::register_converter("f32", &float_to_double);
    double_type::register_converter("rational", &rational_to_double);
}
