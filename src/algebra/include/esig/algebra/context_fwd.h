//
// Created by user on 24/08/22.
//

#ifndef ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_CONTEXT_FWD_H_
#define ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_CONTEXT_FWD_H_

#include <esig/implementation_types.h>
#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/coefficients.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/lie_interface.h>
#include <esig/algebra/tensor_interface.h>

#include <esig/algebra/basis.h>

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

#define ESIG_MAKE_CTYPE_SWITCH(CTYPE)                         \
    switch (CTYPE) {                                          \
        case coefficient_type::dp_real:                       \
            return ESIG_SWITCH_FN(coefficient_type::dp_real); \
        case coefficient_type::sp_real:                       \
            return ESIG_SWITCH_FN(coefficient_type::sp_real); \
    }                                                         \
    throw std::invalid_argument("invalid coefficient_type");

#define ESIG_MAKE_VTYPE_SWITCH_INNER(CTYPE, VTYPE)            \
    switch (CTYPE) {                                           \
        case vector_type::dense:                              \
            return ESIG_SWITCH_FN(CTYPE, vector_type::dense); \
        case vector_type::sparse:                             \
            return ESIG_SWITCH_FN(CTYPE, vector_type::sparse);\
    }                                                         \
    throw std::invalid_argument("invalid vector type");

#define ESIG_MAKE_SWITCH(CTYPE, VTYPE)                               \
    switch (CTYPE) {                                                 \
        case coefficient_type::dp_real:                              \
            ESIG_MAKE_VTYPE_SWITCH_INNER(coefficient_type::dp_real, VTYPE) \
        case coefficient_type::sp_real:                              \
            ESIG_MAKE_VTYPE_SWITCH_INNER(coefficient_type::sp_real, VTYPE) \
    }                                                                \
    throw std::invalid_argument("invalid coefficient_type");




namespace esig {
namespace algebra {

struct data_iterator;

template <typename ToLie>
struct increment_iterable;

struct signature_data;

struct derivative_compute_info;

enum class input_data_type {
    value_array,
    coeff_array,
    key_value_array,
    key_coeff_array
};

class vector_construction_data;

class context;



} // namespace algebra
} // namespace esig


#endif//ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_CONTEXT_FWD_H_
