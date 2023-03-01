//
// Created by user on 02/11/22.
//

#ifndef ESIG_COEFFICIENTS_H_
#define ESIG_COEFFICIENTS_H_

#include "config.h"
#include "esig_export.h"
#include "implementation_types.h"

#include <cassert>
#include <functional>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <boost/variant/variant.hpp>

//namespace esig {
//namespace scalars {
//
//inline std::size_t hash_value(const ScalarType &arg) noexcept { return reinterpret_cast<std::size_t>(&arg); }
//
//using conversion_function = void (*)(void *, const void *, dimn_t);
//
//namespace dtl {
//
//class ESIG_EXPORT scalar_stream_row_iterator;
//
//}// namespace dtl
//
//namespace dtl {
//
//template<typename Scalar>
//class ESIG_EXPORT intermediate_scalar_interface : public ScalarInterface {
//public:
//    virtual Scalar into_value() const noexcept = 0;
//    virtual const Scalar &into_cref() const noexcept = 0;
//};
//
//}// namespace dtl
//

//
//namespace dtl {

// Explicit implementation for float defined in the library

//template <> class ESIG_EXPORT scalar_type_holder<float> {
//   public:
//    static const scalar_type *get_type() noexcept;
//};

// Explicit implementation for double defined in the library.

//template<>
//class ESIG_EXPORT scalar_type_holder<double> {
//public:
//    static const scalar_type *get_type() noexcept;
//};

// explicit instantiation for rational type defined in the library

//template<typename Scalar>
//class scalar_implementation;
;
//
//template<typename Scalar>
//class scalar_implementation : public intermediate_scalar_interface<Scalar> {
//    Scalar m_data;
//    using trait = scalar_type_trait<Scalar>;
//
//    using value_type = typename trait::value_type;
//    using reference = typename trait::reference;
//    using const_reference = typename trait::const_reference;
//
//    scalar_implementation(Scalar s) : m_data(std::move(s)) {}
//public:
//
//
//    const scalar_type *type() const noexcept override {
//        return scalar_type_holder<std::remove_cv_t<Scalar>>::get_type();
//    }
//    bool is_const() const noexcept override {
//        return std::is_const<Scalar>::value;
//    }
//    bool is_value() const noexcept override {
//        return !std::is_reference<Scalar>::value;
//    }
//    bool is_zero() const noexcept override {
//        return m_data == value_type(0);
//    }
//    scalar_t as_scalar() const override {
//        return static_cast<scalar_t>(m_data);
//    }
//
//    scalar uminus() const override {
//        return trait::make(-m_data);
//    }
//    Scalar into_value() const noexcept override {
//        return static_cast<Scalar>(m_data);
//    }
//    const Scalar &into_cref() const noexcept override {
//        return static_cast<const Scalar &>(m_data);
//    }
//
//#define ESIG_SCALAR_GENERATE_DF(NAME, OP)                                                                        \
//    scalar NAME(const scalar_interface *other) const override {                                                  \
//        if (this->type() == other->type()) {                                                                     \
//            const auto &o_ref = static_cast<const typename trait::intermediate_interface *>(other)->into_cref(); \
//            return trait::make(m_data OP o_ref);                                                                 \
//        }                                                                                                        \
//        return trait::make(m_data OP static_cast<value_type>(other->as_scalar()));                               \
//    }
//
//    ESIG_SCALAR_GENERATE_DF(add, +)
//    ESIG_SCALAR_GENERATE_DF(sub, -)
//    ESIG_SCALAR_GENERATE_DF(mul, *)
//
//#undef ESIG_SCALAR_GENERATE_DF
//
//    scalar div(const scalar_interface *other) const override {
//        // Divide is different because we actually want the rational class
//        if (this->type()->rational_type() == other->type()) {
//            const auto &o_ref = static_cast<const typename trait::rational_intermediate_interface *>(other)->into_cref();
//            return trait::make(m_data / o_ref);
//        }
//        return trait::make(m_data / static_cast<value_type>(other->as_scalar()));
//    }
//
//#define ESIG_SCALAR_GENERATE_DFI(NAME, OP)                                                                       \
//    void NAME(const scalar_interface *other) override {                                                \
//        if (this->type() == other->type()) {                                                                     \
//            const auto &o_ref = static_cast<const typename trait::intermediate_interface *>(other)->into_cref(); \
//            m_data OP o_ref;                                                                                   \
//        } else {                                                                                                 \
//            m_data OP static_cast<typename trait::rational_type>(other->as_scalar());                                                 \
//        }                                                                                                        \
//    }
//
//    ESIG_SCALAR_GENERATE_DFI(assign, =)
//    ESIG_SCALAR_GENERATE_DFI(add_inplace, +=)
//    ESIG_SCALAR_GENERATE_DFI(sub_inplace, -=)
//    ESIG_SCALAR_GENERATE_DFI(mul_inplace, *=)
//
//#undef ESIG_SCALAR_GENERATE_DFI
//
//    void div_inplace(const scalar_interface *other) override {
//        if (this->type()->rational_type() == other->type()) {
//            const auto &o_ref = static_cast<const typename trait::rational_intermediate_interface *>(other)->into_cref();
//            m_data /= o_ref;
//        } else {
//            m_data /= static_cast<typename trait::rational_type>(other->as_scalar());
//        }
//    }
//
//
//    bool equals(const scalar_interface *other) const noexcept override {
//        return scalar_interface::equals(other);
//    }
//    std::ostream &print(std::ostream &os) const override {
//        return os << m_data;
//    }
//};

//}// namespace dtl
//}// namespace scalars
//}// namespace esig


#include "scalar_fwd.h"
#include "scalar_pointer.h"
#include "scalar_interface.h"
#include "scalar.h"
#include "scalar_type.h"
#include "scalar_array.h"
#include "owned_scalar_array.h"
#include "key_scalar_array.h"
#include "scalar_stream.h"
#include "scalar_traits.h"



#endif//ESIG_COEFFICIENTS_H_
