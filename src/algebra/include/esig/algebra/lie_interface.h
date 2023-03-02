//
// Created by user on 19/03/2022.
//

#ifndef ESIG_ALGEBRA_LIE_INTERFACE_H_
#define ESIG_ALGEBRA_LIE_INTERFACE_H_


#include <esig/algebra/algebra_traits.h>
#include "esig_algebra_export.h"
#include <esig/algebra/iteration.h>
#include <esig/implementation_types.h>
#include "esig/algebra/algebra_base.h"

#include <ostream>
#include <memory>
#include <type_traits>

namespace esig {
namespace algebra {

class lie;

using lie_interface = AlgebraInterface<lie>;



/**
 * @brief Wrapper class for lie objects
 */
class ESIG_ALGEBRA_EXPORT lie : public AlgebraBase<lie_interface>
{
    using base = AlgebraBase<lie_interface>;

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
