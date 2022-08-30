//
// Created by user on 19/03/2022.
//

#ifndef ESIG_ALGEBRA_LIE_INTERFACE_H_
#define ESIG_ALGEBRA_LIE_INTERFACE_H_


#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/coefficients.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/iteration.h>
#include <esig/implementation_types.h>
#include <esig/algebra/base.h>

#include <iosfwd>
#include <memory>
#include <type_traits>

namespace esig {
namespace algebra {

class lie;

using lie_interface = algebra_interface<lie>;

struct lie_base_access;


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


struct lie_base_access {
    template<typename Impl>
    static const Impl &get(const lie &wrapper)
    {
        auto *ptr = algebra_base_access::get(wrapper);
        assert(ptr != nullptr);
        return algebra_implementation_access::get<dtl::lie_implementation, Impl>(*ptr);
//        return dynamic_cast<const dtl::lie_implementation<Impl> &>(*ptr).m_data;
    }

    template<typename Impl>
    static Impl &get(lie &wrapper)
    {
        auto *ptr = algebra_base_access::get(wrapper);
        assert(ptr != nullptr);
        return algebra_implementation_access::get(*ptr);
//        return dynamic_cast<dtl::lie_implementation<Impl> &>(*ptr).m_data;
    }

//    template <typename Wrapper>
//    static typename Wrapper::impl_type& get(Wrapper& arg)
//    {
//        return arg.m_data;
//    }
//
//    template <typename Wrapper>
//    static const typename Wrapper::impl_type& get(const Wrapper& arg)
//    {
//        return arg.m_data;
//    }


};

ESIG_ALGEBRA_EXPORT
std::ostream &operator<<(std::ostream &os, const lie &arg);

}// namespace algebra
}// namespace esig


#endif//ESIG_ALGEBRA_LIE_INTERFACE_H_
