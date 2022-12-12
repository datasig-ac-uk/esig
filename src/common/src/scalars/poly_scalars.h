//
// Created by user on 22/11/22.
//

#ifndef ESIG_SRC_COMMON_SRC_SCALARS_POLY_SCALARS_H_
#define ESIG_SRC_COMMON_SRC_SCALARS_POLY_SCALARS_H_

#include "standard_scalar_type.h"

#include <libalgebra/libalgebra.h>
#include <libalgebra/coefficients.h>
#include <libalgebra/rational_coefficients.h>

namespace esig {

using poly_scalar_type = alg::poly<alg::coefficients::coefficient_ring<rational_scalar_type>>>;

class poly_scalars : public standard_scalar_type<poly_scalar_type>
{
public:
    poly_scalars();
};

}// namespace esig

#endif//ESIG_SRC_COMMON_SRC_SCALARS_POLY_SCALARS_H_
