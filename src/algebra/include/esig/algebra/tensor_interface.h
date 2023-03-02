//
// Created by user on 17/03/2022.
//

#ifndef ESIG_ALGEBRA_FREE_TENSOR_INTERFACE_H_
#define ESIG_ALGEBRA_FREE_TENSOR_INTERFACE_H_

#include "free_tensor.h"
#include "shuffle_tensor.h"

/*#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/iteration.h>
#include <esig/algebra/context_fwd.h>

#include <memory>
#include <type_traits>
#include <ostream>*/

namespace esig {
namespace algebra {

/*
class free_tensor;
class shuffle_tensor;


extern template class ESIG_ALGEBRA_EXPORT AlgebraInterface<free_tensor>;

class ESIG_ALGEBRA_EXPORT free_tensor_interface : public AlgebraInterface<free_tensor>
{

};



extern template class ESIG_ALGEBRA_EXPORT AlgebraInterface<shuffle_tensor>;


using shuffle_tensor_interface = AlgebraInterface<shuffle_tensor>;




namespace dtl {
*/
/*
 * Typically, the interface will be implemented via the following template wrapper
 * which takes the standard arithmetic operations to implement the arithmetic, and
 * uses traits to identify the properties.
 *//*


template <typename Impl>
class free_tensor_implementation : public algebra_implementation<free_tensor_interface, Impl>
{
    using base = algebra_implementation<free_tensor_interface, Impl>;
    friend class algebra_access<free_tensor_interface>;
    friend class algebra_access<AlgebraInterface<free_tensor>>;

public:

    using base::base;

    free_tensor exp() const override;
    free_tensor log() const override;
    free_tensor inverse() const override;
    free_tensor_interface &fmexp(const free_tensor_interface &other) override;
};

template <typename Impl>
class borrowed_free_tensor_implementation : public borrowed_algebra_implementation<free_tensor_interface, Impl>
{
    using base = borrowed_algebra_implementation<free_tensor_interface, Impl>;
    friend class ::esig::algebra::algebra_access<free_tensor_interface>;
    friend class ::esig::algebra::algebra_access<AlgebraInterface<free_tensor>>;


public:

    using base::base;

    free_tensor exp() const override;
    free_tensor log() const override;
    free_tensor inverse() const override;
    free_tensor_interface &fmexp(const free_tensor_interface &other) override;
};

template <typename Impl>
struct implementation_wrapper_selection<free_tensor_interface, Impl>
{
    using type = std::conditional_t<std::is_base_of<free_tensor_interface, Impl>::value, Impl,
          free_tensor_implementation<Impl>>;
};

template <typename Impl>
struct implementation_wrapper_selection<free_tensor_interface, Impl*>
{
    using type = dtl::borrowed_free_tensor_implementation<Impl>;
};



template <typename Impl>
using shuffle_tensor_implementation = algebra_implementation<shuffle_tensor_interface, Impl>;

} // namespace dtl


extern template class ESIG_ALGEBRA_EXPORT AlgebraBase<free_tensor_interface>;

*/
/**
 * @brief Wrapper class for free tensor objects.
 *//*

class ESIG_ALGEBRA_EXPORT free_tensor : public AlgebraBase<free_tensor_interface>
{
    friend class algebra_base_access;

    using base = AlgebraBase<free_tensor_interface>;

public:

    using base::base;

    free_tensor exp() const;
    free_tensor log() const;
    free_tensor inverse() const;
    free_tensor & fmexp(const free_tensor & other);

};




extern template class ESIG_ALGEBRA_EXPORT AlgebraBase<shuffle_tensor_interface>;

class shuffle_tensor : public AlgebraBase<shuffle_tensor_interface>
{
    using base = AlgebraBase<shuffle_tensor_interface>;
public:
    using base::base;
};
*/


//// We have to define the template constructor here too.
//template<typename Impl, typename>
//free_tensor::free_tensor(Impl &&impl, const context* ctx)
//        : p_impl(new dtl::free_tensor_implementation<
//        typename std::remove_cv<
//                typename std::remove_reference<Impl>::type>::type>(std::forward<Impl>(impl), ctx))
//{
//}
//
//template<typename Impl, typename... Args>
//free_tensor free_tensor::from_args(Args &&...args)
//{
//    std::shared_ptr<free_tensor_interface> p(new Impl(std::forward<Args>(args)...));
//    return free_tensor(p);
//}


// The rest of this file are implementations of the template wrapper methods.




//
//
//} // namespace dtl



} // namespace algebra
} // namespace esig



#endif//ESIG_ALGEBRA_FREE_TENSOR_INTERFACE_H_
