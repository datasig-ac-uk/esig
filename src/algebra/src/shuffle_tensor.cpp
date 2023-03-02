//
// Created by user on 24/08/22.
//
#include "esig/algebra/shuffle_tensor.h"
#include <ostream>

template class esig::algebra::AlgebraInterface<esig::algebra::ShuffleTensor>;
template class esig::algebra::AlgebraBase<esig::algebra::ShuffleTensorInterface>;



std::ostream& operator<<(std::ostream& os, const esig::algebra::ShuffleTensor& arg)
{
    return arg.print(os);
}
