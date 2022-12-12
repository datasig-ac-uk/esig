//
// Created by user on 14/11/22.
//

#include "double_type.h"

using namespace esig;
using namespace scalars;

template <>
const scalar_type* dtl::scalar_type_holder<double>::get_type() noexcept
{
    static const double_type dtype;
    return &dtype;
}
