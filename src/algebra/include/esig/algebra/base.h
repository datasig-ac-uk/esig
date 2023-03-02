//
// Created by user on 23/08/22.
//

#ifndef ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_BASE_H_
#define ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_BASE_H_


#include "algebra_fwd.h"
#include "basis.h"
#include "algebra_info.h"
#include "algebra_base.h"
#include "algebra_bundle.h"
#include "linear_operator.h"
#include "algebra_impl.h"


//#include <esig/implementation_types.h>
//
//#include <algorithm>
//#include <cassert>
//#include <memory>
//#include <iosfwd>
//#include <type_traits>
//#include <utility>
//#include <vector>
//
//#include <boost/optional.hpp>
//#include <boost/type_traits/copy_cv.hpp>
//#include <boost/type_traits/copy_cv_ref.hpp>
//#include <boost/type_traits/is_detected.hpp>
//
//#include <esig/scalars.h>
//#include "esig/algebra/esig_algebra_export.h"
//#include "algebra_traits.h"
//#include "iteration.h"


/*
namespace esig {
namespace algebra {


template <typename Algebra>
struct algebra_info;




namespace dtl {


} // namespace dtl









template <typename Argument, typename Result=Argument>
struct OperatorInterface {
    using argument_type = Argument;
    using result_type = Result;

    virtual ~OperatorInterface() = default;

    virtual result_type apply(const argument_type& argument) const = 0;
};



namespace dtl {





template <typename AlgebraInterface,
         typename Impl,
         typename FibreInterface,
         template <typename, typename> class AlgebraWrapper=algebra_implementation,
         template <typename, typename> class FibreWrapper=borrowed_algebra_implementation>
class algebra_bundle_implementation
    : public AlgebraWrapper<AlgebraBundleInterface<AlgebraInterface,
                                 typename FibreInterface::algebra_t>, Impl>
{
    using wrapped_alg_interface = AlgebraBundleInterface<
        AlgebraInterface, typename FibreInterface::algebra_t>;

    using fibre_impl_type = decltype(std::declval<Impl>().fibre());
    using base = AlgebraWrapper<wrapped_alg_interface, Impl>;

    using fibre_wrap_type = FibreWrapper<FibreInterface, fibre_impl_type>;

    using alg_info = algebra_info<Impl>;
    using fibre_info = algebra_info<fibre_impl_type>;

public:
    using fibre_type = typename FibreInterface::algebra_t;
    using fibre_interface_t = FibreInterface;
    using fibre_scalar_type = typename algebra_info<fibre_impl_type>::scalar_type;
    using fibre_rational_type = typename algebra_info<fibre_impl_type>::rational_type;

    using base::base;

    fibre_type fibre() override;
};


template <typename Interface, typename Impl>
class operator_implementation : public Interface
{
    using base_interface = OperatorInterface<
        typename Interface::argument_type,
        typename Interface::result_type>;

    static_assert(std::is_base_of<base_interface, Interface>::value,
                  "Interface must be derived from operator_interface");

    Impl m_impl;
    std::shared_ptr<const context> p_ctx;

public:
    using argument_type = typename base_interface::argument_type;
    using result_type = typename base_interface::result_type;

    using interface_t = Interface;

    explicit operator_implementation(Impl&& arg, std::shared_ptr<const context> ctx)
        : m_impl(std::move(arg)), p_ctx(std::move(ctx))
    {}

    result_type apply(const argument_type& arg) const
    {
        return result_type(m_impl(arg), p_ctx.get());
    }

};

template<typename T>
using impl_options = std::conditional_t<std::is_pointer<T>::value, std::true_type, std::false_type>;

template <typename Interface, typename Impl, typename Options=impl_options<Impl>>
struct implementation_wrapper_selection
{
    static_assert(!std::is_pointer<Impl>::value, "Impl cannot be a pointer");
    using type = typename std::conditional<std::is_base_of<Interface, Impl>::value, Impl,
        algebra_implementation<Interface, Impl>>::type;
};

template <typename Interface, typename Impl>
struct implementation_wrapper_selection<Interface, Impl, std::true_type>
{
    using type = borrowed_algebra_implementation<Interface, Impl>;
};

template <typename BaseInterface, typename Fibre, typename Impl, typename Options>
struct implementation_wrapper_selection<
    AlgebraBundleInterface<BaseInterface, Fibre>,
    Impl, Options>
{
    using type = algebra_bundle_implementation<BaseInterface, Impl, AlgebraInterface<Fibre>>;
};

template <typename Interface, typename Impl>
using impl_wrapper = typename implementation_wrapper_selection<Interface, std::remove_pointer_t<Impl>, impl_options<Impl>>::type;

} // namespace dtl




template <typename Interface>
class LinearOperatorBase {
    std::shared_ptr<const Interface> p_impl;
public:

    using argument_type = typename Interface::argument_type;
    using result_type = typename Interface::argument_type;

    result_type operator()(const argument_type& arg) const
    {
        return p_impl->apply(arg);
    }

};

/////////////////////////////////////////////////////////////////////////
// The rest of the file contains implementations for the templates above.
/////////////////////////////////////////////////////////////////////////

// First up is the algebra_info trait, we'll need this throughout the rest
// of the implementations.







namespace dtl {






template<typename AlgebraInterface,
         typename Impl,
         typename FibreInterface,
         template<typename, typename> class AlgebraWrapper,
         template<typename, typename> class FibreWrapper>
typename algebra_bundle_implementation<AlgebraInterface, Impl, FibreInterface, AlgebraWrapper, FibreWrapper>::fibre_type
algebra_bundle_implementation<AlgebraInterface, Impl, FibreInterface, AlgebraWrapper, FibreWrapper>::fibre() {
    return fibre_type(fibre_wrap_type(&base::m_data.fibre()), base::p_ctx);
}

} // namespace dtl




}
//template<typename Interface>
//dense_data_access_iterator algebra_base<Interface>::iterate_dense_components() const noexcept {
//    return p_impl->iterate_dense_components();
//}

//template<typename Interface>
//typename algebra_base<Interface>::algebra_t algebra_base<Interface>::left_smul(const coefficient &scal) const {
//    return p_impl->left_smul(scal);
//}
//template<typename Interface>
//typename algebra_base<Interface>::algebra_t algebra_base<Interface>::right_smul(const coefficient &scal) const {
//    return p_impl->right_smul(scal);
//}


//template<typename Interface>
//algebra_base<Interface> &algebra_base<Interface>::left_smul_inplace(const coefficient &rhs) {
//    p_impl->left_smul_inplace(rhs);
//    return *this;
//}
//template<typename Interface>
//algebra_base<Interface> &algebra_base<Interface>::right_smul_inplace(const coefficient &rhs) {
//    p_impl->right_smul_inplace(rhs);
//    return *this;
//}


//template<typename Interface>
//algebra_base<Interface> &algebra_base<Interface>::mul_left_smul(const algebra_base<Interface> &arg, const coefficient &scal) {
//    p_impl->mul_left_smul(*arg.p_impl, scal);
//    return *this;
//}
//template<typename Interface>
//algebra_base<Interface> &algebra_base<Interface>::mul_right_smul(const algebra_base<Interface> &arg, const coefficient &scal) {
//    p_impl->mul_right_smul(*arg.p_impl, scal);
//    return *this;
//}








template <typename Interface>
struct algebra_access
{

    using interface_type = Interface;

    template <typename Impl>
    using wrapper_t = typename dtl::implementation_wrapper_selection<Interface, Impl, std::false_type>::type;

    template <typename Impl>
    using borrowed_wrapper_t = typename dtl::implementation_wrapper_selection<Interface, Impl, std::true_type>::type;

    template <typename Impl>
    static const Impl& get(const interface_type& arg)
    {
        if (arg.type() == ImplementationType::owned) {
            return static_cast<const wrapper_t<Impl>&>(arg).m_data;
        } else {
            return *static_cast<const borrowed_wrapper_t<Impl>&>(arg).p_impl;
        }
    }

    template <typename Impl>
    static Impl& get(interface_type& arg)
    {
        if (arg.type() == ImplementationType::owned) {
            return static_cast<wrapper_t<Impl>&>(arg).m_data;
        } else {
            return *static_cast<borrowed_wrapper_t<Impl>&>(arg).p_impl;
        }
    }

    template <typename Impl>
    static Impl& get(typename interface_type::algebra_t& arg)
    {
        return get<Impl>(*arg.p_impl);
    }

    template <typename Impl>
    static const Impl& get(const typename interface_type::algebra_t& arg)
    {
        return get<Impl>(*arg.p_impl);
    }

    static const interface_type* get(const typename interface_type::algebra_t& arg)
    {
        return arg.p_impl.get();
    }
    static interface_type* get(typename interface_type::algebra_t& arg)
    {
        return arg.p_impl.get();
    }

};




} // namespace algebra
} // namespace esig
*/


#endif//ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_BASE_H_
