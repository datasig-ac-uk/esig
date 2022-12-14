//
// Created by user on 17/03/2022.
//

#ifndef ESIG_ALGEBRA_FREE_TENSOR_INTERFACE_H_
#define ESIG_ALGEBRA_FREE_TENSOR_INTERFACE_H_


#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/iteration.h>
#include <esig/algebra/context_fwd.h>

#include <memory>
#include <type_traits>
#include <iosfwd>

namespace esig {
namespace algebra {

class free_tensor;
class shuffle_tensor;


extern template class ESIG_ALGEBRA_EXPORT algebra_interface<free_tensor>;

class ESIG_ALGEBRA_EXPORT free_tensor_interface : public algebra_interface<free_tensor>
{
public:
    // Special functions
    virtual free_tensor exp() const = 0;
    virtual free_tensor log() const = 0;
    virtual free_tensor inverse() const = 0;
    virtual free_tensor_interface& fmexp(const free_tensor_interface& other) = 0;
};



extern template class ESIG_ALGEBRA_EXPORT algebra_interface<shuffle_tensor>;


using shuffle_tensor_interface = algebra_interface<shuffle_tensor>;




namespace dtl {
/*
 * Typically, the interface will be implemented via the following template wrapper
 * which takes the standard arithmetic operations to implement the arithmetic, and
 * uses traits to identify the properties.
 */

template <typename Impl>
class free_tensor_implementation : public algebra_implementation<free_tensor_interface, Impl>
{
    using base = algebra_implementation<free_tensor_interface, Impl>;
    friend class algebra_access<free_tensor_interface>;
    friend class algebra_access<algebra_interface<free_tensor>>;

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
    friend class ::esig::algebra::algebra_access<algebra_interface<free_tensor>>;


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


extern template class ESIG_ALGEBRA_EXPORT algebra_base<free_tensor_interface>;

/**
 * @brief Wrapper class for free tensor objects.
 */
class ESIG_ALGEBRA_EXPORT free_tensor : public algebra_base<free_tensor_interface>
{
    friend class algebra_base_access;

    using base = algebra_base<free_tensor_interface>;

public:

    using base::base;

    free_tensor exp() const;
    free_tensor log() const;
    free_tensor inverse() const;
    free_tensor & fmexp(const free_tensor & other);

};

inline std::ostream& operator<<(std::ostream& os, const free_tensor& ft)
{
    return os << static_cast<const algebra_base<free_tensor_interface>&>(ft);
}


extern template class ESIG_ALGEBRA_EXPORT algebra_base<shuffle_tensor_interface>;

class shuffle_tensor : public algebra_base<shuffle_tensor_interface>
{
    using base = algebra_base<shuffle_tensor_interface>;
public:
    using base::base;
};




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


namespace dtl {

template <typename Tensor>
Tensor exp_wrapper(const Tensor& arg)
{
    return exp(arg);
}

template <typename Tensor>
Tensor log_wrapper(const Tensor& arg)
{
    return log(arg);
}

template <typename Tensor>
Tensor inverse_wrapper(const Tensor& arg)
{
    return inverse(arg);
}


template<typename Impl>
free_tensor free_tensor_implementation<Impl>::exp() const
{
    return free_tensor(exp_wrapper(free_tensor_implementation::m_data), free_tensor_implementation::p_ctx);
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::log() const
{
    return free_tensor(log_wrapper(free_tensor_implementation::m_data), free_tensor_implementation::p_ctx);
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::inverse() const
{
    return free_tensor(inverse_wrapper(free_tensor_implementation::m_data), free_tensor_implementation::p_ctx);
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::fmexp(const free_tensor_interface &other)
{
    free_tensor_implementation::m_data.fmexp_inplace(dynamic_cast<const free_tensor_implementation&>(other).m_data);
    return *this;
}
template<typename Impl>
free_tensor borrowed_free_tensor_implementation<Impl>::exp() const {
    return free_tensor(exp_wrapper(*borrowed_free_tensor_implementation::p_impl), borrowed_free_tensor_implementation::p_ctx);
}
template<typename Impl>
free_tensor borrowed_free_tensor_implementation<Impl>::log() const {
    return free_tensor(log_wrapper(*borrowed_free_tensor_implementation::p_impl), borrowed_free_tensor_implementation::p_ctx);
}
template<typename Impl>
free_tensor borrowed_free_tensor_implementation<Impl>::inverse() const {
    return free_tensor(inverse_wrapper(*borrowed_free_tensor_implementation::p_impl), borrowed_free_tensor_implementation::p_ctx);
}
template<typename Impl>
free_tensor_interface &borrowed_free_tensor_implementation<Impl>::fmexp(const free_tensor_interface &other) {
    base::p_impl->fmexp(base::cast(other));
    return *this;
}

} // namespace dtl



} // namespace algebra
} // namespace esig



#endif//ESIG_ALGEBRA_FREE_TENSOR_INTERFACE_H_
