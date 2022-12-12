//
// Created by user on 24/08/22.
//

#ifndef ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_CONTEXT_FWD_H_
#define ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_CONTEXT_FWD_H_

#include <esig/implementation_types.h>
#include "algebra_traits.h"
#include <esig/algebra/esig_algebra_export.h>
#include "lie_interface.h"
#include "tensor_interface.h"

#include "basis.h"

#include <stdexcept>
#include <vector>

#define ESIG_MAKE_VTYPE_SWITCH(VTYPE)                          \
    switch (VTYPE) {                                           \
        case vector_type::dense:                               \
            return ESIG_SWITCH_FN(vector_type::dense);         \
        case vector_type::sparse:                              \
            return ESIG_SWITCH_FN(vector_type::sparse);        \
    }                                                          \
    throw std::invalid_argument("invalid vector type");



namespace esig {
namespace algebra {

struct signature_data;
struct derivative_compute_info;
class vector_construction_data;
class context;


} // namespace algebra
} // namespace esig


#endif//ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_CONTEXT_FWD_H_
