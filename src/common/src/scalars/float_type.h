//
// Created by user on 14/11/22.
//

#ifndef ESIG_SRC_COMMON_SRC_SCALARS_FLOAT_TYPE_H_
#define ESIG_SRC_COMMON_SRC_SCALARS_FLOAT_TYPE_H_

#include "standard_scalar_type.h"


namespace esig {
namespace scalars {

class float_type : public standard_scalar_type<float> {
public:
    float_type();
};

} // namespace scalars
}// namespace esig

#endif//ESIG_SRC_COMMON_SRC_SCALARS_FLOAT_TYPE_H_
