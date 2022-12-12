//
// Created by user on 14/11/22.
//

#include "float_type.h"

using namespace esig;
using namespace scalars;

static void double_to_float(void* dst, const void* src)
{
    ::new (dst) float(*static_cast<const double*>(src));
}

static void rational_to_float(void* dst, const void* src)
{
    ::new (dst) float(*static_cast<const rational_scalar_type*>(src));
}

float_type::float_type() : standard_scalar_type<float>("f32", "float") {
    standard_scalar_type<float>::register_converter("f64", &double_to_float);
    float_type::register_converter("rational", &rational_to_float);
}
