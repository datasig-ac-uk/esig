//
// Created by user on 19/03/2022.
//

#ifndef ESIG_ALGEBRA_LIE_INTERFACE_H_
#define ESIG_ALGEBRA_LIE_INTERFACE_H_


#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/iteration.h>
#include <esig/implementation_types.h>
#include <esig/algebra/base.h>

#include <ostream>
#include <memory>
#include <type_traits>

namespace esig {
namespace algebra {

class lie;

using lie_interface = algebra_interface<lie>;


namespace dtl {
/*
 * Typically, the interface will be implemented via the following template wrapper
 * which takes the standard arithmetic operations to implement the arithmetic, and
 * uses traits to identify the properties.
 */

template <typename Impl>
using lie_implementation = algebra_implementation<lie_interface, Impl>;

template <typename Impl>
struct implementation_wrapper_selection<lie_interface, Impl>
{
    using type = lie_implementation<Impl>;
};


}// namespace dtl


/**
 * @brief Wrapper class for lie objects
 */
class ESIG_ALGEBRA_EXPORT lie : public algebra_base<lie_interface>
{
    using base = algebra_base<lie_interface>;

public:
    using base::base;

};

//
//inline std::ostream& operator<<(std::ostream& os, const lie& arg)
//{
//    return os << static_cast<const algebra_base<lie_interface>&>(arg);
//}


}// namespace algebra
}// namespace esig


#endif//ESIG_ALGEBRA_LIE_INTERFACE_H_
