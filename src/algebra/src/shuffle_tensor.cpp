//
// Created by user on 24/08/22.
//
#include <esig/algebra/tensor_interface.h>
#include <iostream>

template class esig::algebra::AlgebraInterface<esig::algebra::shuffle_tensor>;
template class esig::algebra::algebra_base<esig::algebra::shuffle_tensor_interface>;



std::ostream& operator<<(std::ostream& os, const esig::algebra::shuffle_tensor& arg)
{
    return arg.print(os);
}
