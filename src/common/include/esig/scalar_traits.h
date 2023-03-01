#ifndef ESIG_COMMON_SCALAR_TRAITS_H_
#define ESIG_COMMON_SCALAR_TRAITS_H_

#include "implementation_types.h"

#include "scalar.h"
#include "scalar_fwd.h"
#include "scalar_type.h"

namespace esig {
namespace scalars {

namespace dtl {

template<typename ScalarImpl>
struct scalar_type_holder<ScalarImpl &> : scalar_type_holder<ScalarImpl> {};

template<typename ScalarImpl>
struct scalar_type_holder<const ScalarImpl &> : scalar_type_holder<ScalarImpl> {};

template<>
ESIG_EXPORT const ScalarType *scalar_type_holder<float>::get_type() noexcept;

template<>
ESIG_EXPORT const ScalarType *scalar_type_holder<double>::get_type() noexcept;

template<>
ESIG_EXPORT const ScalarType *scalar_type_holder<rational_scalar_type>::get_type() noexcept;

template<typename T>
class scalar_type_trait {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T &;
    using const_reference = const T &;

    static const ScalarType *get_type() noexcept {
        return scalar_type_holder<T>::get_type();
    }

    static Scalar make(value_type &&arg) {
        return Scalar(arg, get_type());
    }
};

template<typename T>
class scalar_type_trait<T &> {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T &;
    using const_reference = const T &;

    static const ScalarType *get_type() noexcept {
        return scalar_type_trait<T>::get_type();
    }

    static Scalar make(reference arg) {
        return Scalar(ScalarPointer(&arg, get_type()));
    }
};

template<typename T>
class scalar_type_trait<const T &> {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T &;
    using const_reference = const T &;

    static const ScalarType *get_type() noexcept {
        return scalar_type_trait<T>::get_type();
    }

    static Scalar make(const_reference arg) {
        return Scalar(ScalarPointer(&arg, get_type()));
    }
};

#define ESIG_MAKE_TYPE_ID_OF(TYPE, NAME)              \
    template<>                                        \
    struct type_id_of_impl<TYPE> {                    \
        static const std::string &get_id() noexcept { \
            static const std::string type_id(NAME);   \
            return type_id;                           \
        }                                             \
    }

ESIG_MAKE_TYPE_ID_OF(char, "i8");
ESIG_MAKE_TYPE_ID_OF(unsigned char, "u8");
ESIG_MAKE_TYPE_ID_OF(short, "i16");
ESIG_MAKE_TYPE_ID_OF(unsigned short, "u16");
ESIG_MAKE_TYPE_ID_OF(int, "i32");
ESIG_MAKE_TYPE_ID_OF(unsigned int, "u32");
ESIG_MAKE_TYPE_ID_OF(long long, "i64");
ESIG_MAKE_TYPE_ID_OF(unsigned long long, "u64");
ESIG_MAKE_TYPE_ID_OF(signed_size_type_marker, "isize");
ESIG_MAKE_TYPE_ID_OF(unsigned_size_type_marker, "usize");

ESIG_MAKE_TYPE_ID_OF(float, "f32");
ESIG_MAKE_TYPE_ID_OF(double, "f64");

// Long is silly. On Win64 it is 32 bits (because, Microsoft) on Unix, it is 64 bits
template<>
struct type_id_of_impl<long>
    : public std::conditional_t<(sizeof(long) == sizeof(int)),
                                type_id_of_impl<int>,
                                type_id_of_impl<long long>> {};

#undef ESIG_MAKE_TYPE_ID_OF

}// namespace dtl

template<typename T>
std::remove_cv_t<std::remove_reference_t<T>>
scalar_cast(const Scalar &arg) {
    using trait = dtl::scalar_type_trait<T>;
    const auto *type = arg.type();

    if (type == trait::get_type()) {
        return T(*arg.to_const_pointer().template raw_cast<const T>());
    }
    if (type->rational_type() == trait::get_type()) {
        using R = typename trait::rational_type;
        return T(*arg.to_const_pointer().template raw_cast<const R>());
    }

    return T(arg.to_scalar_t());
}


}// namespace scalars
}// namespace esig

#endif// ESIG_COMMON_SCALAR_TRAITS_H_
