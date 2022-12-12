//
// Created by user on 22/11/22.
//

#ifndef ESIG_SRC_COMMON_SRC_SCALARS_RATIONAL_TYPE_H_
#define ESIG_SRC_COMMON_SRC_SCALARS_RATIONAL_TYPE_H_

#include "standard_scalar_type.h"


namespace esig {
namespace scalars {

class rational_type : public standard_scalar_type<rational_scalar_type> {
public:
    rational_type();
};

} // namespace scalars
}// namespace esig

#endif//ESIG_SRC_COMMON_SRC_SCALARS_RATIONAL_TYPE_H_
