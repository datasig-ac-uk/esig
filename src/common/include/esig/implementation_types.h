//
// Created by user on 20/03/2022.
//

#ifndef ESIG_IMPLEMENTATION_TYPES_H_
#define ESIG_IMPLEMENTATION_TYPES_H_

#include <cstdint>


namespace esig {
using deg_t = unsigned;
using dimn_t = std::size_t;
using idimn_t = std::ptrdiff_t;
using let_t = unsigned;
using key_type = std::size_t;


using scalar_t = double;// Python floating-points are doubles.
using param_t = double;
using accuracy_t = double;



} // namespace esig

#endif//ESIG_IMPLEMENTATION_TYPES_H_
