//
// Created by user on 02/11/22.
//

#ifndef ESIG_COEFFICIENTS_H_
#define ESIG_COEFFICIENTS_H_

#include "esig_export.h"
#include "implementation_types.h"
#include "config.h"

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


namespace esig {
namespace scalars {

struct scalar_type_info {
    std::string id;
    std::string name;
    int size;
    int alignment;
};

class scalar_type;
class scalar;
class scalar_pointer;
class scalar_array;
class scalar_stream;
class owned_scalar_array;

namespace dtl {
template<typename T>
class scalar_type_trait;
}// namespace dtl

class ESIG_EXPORT scalar_interface {
    friend class scalar;

public:
    virtual ~scalar_interface() = default;

    virtual const scalar_type *type() const noexcept = 0;

    virtual bool is_const() const noexcept = 0;
    virtual bool is_value() const noexcept = 0;
    virtual bool is_zero() const noexcept = 0;

    virtual scalar_t as_scalar() const = 0;
    virtual void assign(scalar_pointer) = 0;
    virtual void assign(const scalar& other) = 0;

    virtual scalar_pointer to_pointer() = 0;
    virtual scalar_pointer to_pointer() const noexcept = 0;
    virtual scalar uminus() const;

//    virtual scalar add(const scalar &other) const;
//    virtual scalar sub(const scalar &other) const;
//    virtual scalar mul(const scalar &other) const;
//    virtual scalar div(const scalar &other) const;

    virtual void add_inplace(const scalar &other) = 0;
    virtual void sub_inplace(const scalar &other) = 0;
    virtual void mul_inplace(const scalar &other) = 0;
    virtual void div_inplace(const scalar &other) = 0;

    virtual bool equals(const scalar &other) const noexcept;

    virtual std::ostream &print(std::ostream &os) const;
};

class ESIG_EXPORT scalar_type {
    const scalar_type_info m_info;

    friend class scalar_pointer;
    friend class scalar;

protected:
    std::pair<scalar, void *> new_owned_scalar() const;

public:
    using converter_function = void (*)(void *, const void *);
    virtual void register_converter(const std::string &id, converter_function func) const;
    virtual converter_function get_converter(const std::string &id) const noexcept;

    explicit scalar_type(scalar_type_info info)
        : m_info(std::move(info)) {}

    // No copy/move constructors or assigments
    scalar_type(const scalar_type &) = delete;
    scalar_type(scalar_type &&) noexcept = delete;
    scalar_type &operator=(const scalar_type &) = delete;
    scalar_type &operator=(scalar_type &&) noexcept = delete;

    virtual ~scalar_type();

    virtual scalar from(int) const = 0;
    virtual scalar from(long long, long long) const = 0;

    const std::string &id() const noexcept { return m_info.id; }
    const scalar_type_info &info() const noexcept { return m_info; }
    int alignment() const noexcept { return m_info.alignment; }
    int itemsize() const noexcept { return m_info.size; }

    virtual const scalar_type *rational_type() const noexcept;

    virtual scalar_pointer allocate(dimn_t) const = 0;
    virtual void deallocate(scalar_pointer, dimn_t) const = 0;
    virtual void convert_copy(void *out, scalar_pointer in, dimn_t count) const = 0;
    virtual void convert_copy(scalar_pointer &out, const void *in, dimn_t count, const std::string &type_id) const = 0;
    virtual scalar convert(scalar_pointer other) const = 0;

    virtual scalar one() const;
    virtual scalar mone() const;
    virtual scalar zero() const;

    virtual scalar_t to_scalar_t(const void *arg) const = 0;

    virtual void assign(void *dst, scalar_pointer src) const = 0;
    virtual void assign(void *dst, scalar_pointer src, dimn_t count) const;
    virtual void assign(void *dst, long long numerator, long long denominator) const;
    virtual scalar copy(scalar_pointer arg) const = 0;
    virtual scalar uminus(scalar_pointer arg) const = 0;
    virtual scalar add(const void *lhs, scalar_pointer rhs) const = 0;
    virtual scalar sub(const void *lhs, scalar_pointer rhs) const = 0;
    virtual scalar mul(const void *lhs, scalar_pointer rhs) const = 0;
    virtual scalar div(const void *lhs, scalar_pointer rhs) const = 0;

    virtual void add_inplace(void *lhs, scalar_pointer rhs) const;
    virtual void sub_inplace(void *lhs, scalar_pointer rhs) const;
    virtual void mul_inplace(void *lhs, scalar_pointer rhs) const;
    virtual void div_inplace(void *lhs, scalar_pointer rhs) const;

    virtual bool is_zero(const void *) const;
    virtual bool are_equal(const void *, const scalar_pointer &rhs) const noexcept = 0;

    virtual void print(const void *, std::ostream &os) const;
};

ESIG_EXPORT void register_type(const std::string &identifier, const scalar_type *type);
ESIG_EXPORT const scalar_type *get_type(const std::string &identifier);

ESIG_EXPORT
inline bool operator==(const scalar_type &lhs, const scalar_type &rhs) noexcept { return &lhs == &rhs; }

ESIG_EXPORT
inline bool operator!=(const scalar_type &lhs, const scalar_type &rhs) noexcept { return &lhs == &rhs; }

inline std::size_t hash_value(const scalar_type &arg) noexcept { return reinterpret_cast<std::size_t>(&arg); }

class ESIG_EXPORT scalar_pointer {

    friend class scalar_type;
    friend class scalar_array;
    friend class scalar_stream;

protected:
    const void *p_data;
    const scalar_type *p_type;

    enum constness {
        IsConst,
        IsMutable
    } m_constness = IsConst;

    scalar_pointer(const void *, const scalar_type *, constness);

public:
    const scalar_type *type() const noexcept { return p_type; }

    scalar_pointer() : p_data(nullptr), p_type(nullptr), m_constness(IsMutable) {}
    explicit scalar_pointer(const scalar_type *type) : p_data(nullptr), p_type(type) {}

    scalar_pointer(void *ptr, const scalar_type *type)
        : p_data(ptr), p_type(type), m_constness(IsMutable) {}
    scalar_pointer(const void *ptr, const scalar_type *type)
        : p_data(ptr), p_type(type), m_constness(IsConst) {}

    template <typename T>
    explicit scalar_pointer(T* ptr)
        : p_data(ptr), p_type(dtl::scalar_type_trait<T>::get_type()), m_constness(IsMutable)
    {}
    template <typename T>
    explicit scalar_pointer(const T* ptr)
        : p_data(ptr), p_type(dtl::scalar_type_trait<T>::get_type()), m_constness(IsConst)
    {}

    const void *ptr() const noexcept { return p_data; }
    void *ptr() noexcept { return const_cast<void *>(p_data); }

    template<typename T>
    const T *raw_cast() const noexcept {
        return static_cast<const T *>(p_data);
    }

    bool is_null() const noexcept { return p_data == nullptr; }
    bool is_const() const noexcept { return m_constness == IsConst; }

    scalar deref();
    scalar deref_mut();

    scalar operator*();
    scalar operator*() const noexcept;

    scalar_pointer operator+(dimn_t index) const noexcept;
    scalar_pointer &operator+=(dimn_t index) noexcept;

    scalar operator[](dimn_t index) const noexcept;
    scalar operator[](dimn_t index);

    bool operator==(const scalar_pointer &other) const noexcept { return p_type == other.p_type && p_data == other.p_data; }
    bool operator!=(const scalar_pointer &other) const noexcept { return p_type != other.p_type || p_data != other.p_data; }
};

ESIG_EXPORT
std::ostream &operator<<(std::ostream &os, const scalar &arg);

class ESIG_EXPORT scalar : private scalar_pointer {
    friend class scalar_interface;
    friend class scalar_pointer;
    friend class scalar_type;

    template <typename T>
    friend class dtl::scalar_type_trait;

    //    const void* p_impl;
    //    const scalar_type* p_type;

    //    enum constness {
    //        IsConst,
    //        IsMutable
    //    } m_constness = IsConst;

    using scalar_pointer::IsConst;
    using scalar_pointer::IsMutable;
    using scalar_pointer::m_constness;

    struct interface_pointer {};

    enum pointer_type {
        OwnedPointer,   // A raw pointer to a scalar, onwed by this
        BorrowedPointer,// A raw pointer to a scalar, borrowed fomr elsewhere
        InterfacePointer// A pointer to a scalar_interface type
    } m_pointer_type = OwnedPointer;

    explicit scalar(scalar_interface *other, interface_pointer);

    scalar(scalar_pointer data, pointer_type ptype);

public:
    scalar() = default;
    explicit scalar(const scalar_type *);
    explicit scalar(scalar_t);
    scalar(scalar_t, const scalar_type *);
    explicit scalar(scalar_pointer ptr);
    explicit scalar(scalar_interface* other);

    template<typename I,
             typename J,
             typename = std::enable_if_t<
                 std::is_integral<I>::value && std::is_integral<J>::value>>
    scalar(I numerator, J denominator, const scalar_type *type)
        : scalar_pointer((type == nullptr ? get_type("rational") : type)->allocate(1)) {
        p_type->assign(const_cast<void *>(p_data),
                       static_cast<long long>(numerator),
                       static_cast<long long>(denominator));
    }

    template<typename Scalar>
    scalar(Scalar arg, const scalar_type *type)
        : scalar_pointer((type == nullptr ? dtl::scalar_type_trait<Scalar>::get_type() : type)->allocate(1)),
          m_pointer_type(OwnedPointer) {
        const auto *arg_type = dtl::scalar_type_trait<Scalar>::get_type();
        p_type->assign(const_cast<void *>(p_data), {std::addressof(arg), arg_type});
    }

    scalar(const scalar &other);
    scalar(scalar &&other) noexcept;

    ~scalar();

    scalar &operator=(const scalar &other);
    scalar &operator=(scalar &&other) noexcept;

    using scalar_pointer::is_const;
    //    bool is_const() const noexcept;
    bool is_value() const noexcept;
    bool is_zero() const noexcept;

    using scalar_pointer::type;
    //    const scalar_type *type() const noexcept;

    scalar_pointer to_pointer();
    scalar_pointer to_const_pointer() const;
    void set_to_zero();

    scalar_t to_scalar_t() const;

    scalar operator-() const;

    scalar operator+(const scalar &other) const;
    scalar operator-(const scalar &other) const;
    scalar operator*(const scalar &other) const;
    scalar operator/(const scalar &other) const;

    scalar &operator+=(const scalar &other);
    scalar &operator-=(const scalar &other);
    scalar &operator*=(const scalar &other);
    scalar &operator/=(const scalar &other);

    bool operator==(const scalar &rhs) const noexcept;
    bool operator!=(const scalar &rhs) const noexcept;

    friend std::ostream &operator<<(std::ostream &os, const scalar &arg);
};


ESIG_EXPORT std::ostream& operator<<(std::ostream& os, const scalar& arg);

class scalar_array : public scalar_pointer {
    friend class scalar_stream;

protected:
    dimn_t m_size;

public:
    scalar_array() : scalar_pointer(), m_size(0) {}

    scalar_array(const scalar_array &other) noexcept;
    scalar_array(scalar_array &&other) noexcept;

    explicit scalar_array(const scalar_type *type);
    scalar_array(void *data, const scalar_type *type, dimn_t size);
    scalar_array(const void *data, const scalar_type *type, dimn_t size);
    scalar_array(scalar_pointer begin, dimn_t size)
        : scalar_pointer(begin), m_size(size) {}

    scalar_array &operator=(const scalar_array &other) noexcept;
    scalar_array &operator=(scalar_array &&other) noexcept;

    template<typename Int>
    scalar operator[](Int index) {
        auto uindex = static_cast<dimn_t>(index);
        assert(0 <= uindex && uindex < m_size);
        return scalar_pointer::operator[](uindex);
    }

    template<typename Int>
    scalar operator[](Int index) const {
        auto uindex = static_cast<dimn_t>(index);
        assert(0 <= uindex && uindex < m_size);
        return scalar_pointer::operator[](uindex);
    }

    constexpr dimn_t size() const noexcept { return m_size; }
};

class ESIG_EXPORT owned_scalar_array : public scalar_array {
public:
    owned_scalar_array() = default;

    owned_scalar_array(owned_scalar_array &&other) noexcept;

    explicit owned_scalar_array(const scalar_type *type);
    owned_scalar_array(const scalar_type *type, dimn_t size);
    owned_scalar_array(const scalar &value, dimn_t count);
    explicit owned_scalar_array(const scalar_array &other);

    owned_scalar_array &operator=(const scalar_array &other);
    owned_scalar_array &operator=(owned_scalar_array &&other) noexcept;

    ~owned_scalar_array();
};


class ESIG_EXPORT key_scalar_array : public scalar_array {

    const key_type* p_keys = nullptr;
    bool m_scalars_owned = false;
    bool m_keys_owned = true;

public:

    key_scalar_array() = default;
    ~key_scalar_array();

    key_scalar_array(const key_scalar_array& other);
    key_scalar_array(key_scalar_array&& other) noexcept;

    explicit key_scalar_array(owned_scalar_array&& sa) noexcept;
    key_scalar_array(scalar_array base, const key_type* keys);

    explicit key_scalar_array(const scalar_type* type) noexcept;
    explicit key_scalar_array(const scalar_type* type, dimn_t n) noexcept;
    key_scalar_array(const scalar_type *type, const void *begin, dimn_t count) noexcept;

    key_scalar_array& operator=(key_scalar_array&& other) noexcept;
    key_scalar_array& operator=(owned_scalar_array&& other) noexcept;

    const key_type* keys() const noexcept { return p_keys; }
    key_type* keys();
    bool has_keys() const noexcept { return p_keys == nullptr; }

    void allocate_scalars(idimn_t count = -1);
    void allocate_keys(idimn_t count = -1);


};


namespace dtl {

class ESIG_EXPORT scalar_stream_row_iterator;

}// namespace dtl

class ESIG_EXPORT scalar_stream {
    std::vector<const void *> m_stream;
    boost::container::small_vector<dimn_t, 1> m_elts_per_row;
    const scalar_type *p_type;

public:
    const scalar_type *type() const noexcept { return p_type; }

    using const_iterator = dtl::scalar_stream_row_iterator;

    scalar_stream();

    explicit scalar_stream(const scalar_type *type);
    scalar_stream(scalar_pointer base, std::vector<dimn_t> shape);

    scalar_stream(std::vector<const void *> &&stream, dimn_t row_elts, const scalar_type *type)
        : m_stream(stream), m_elts_per_row{row_elts}, p_type(type) {}

    dimn_t col_count(dimn_t i = 0) const noexcept;
    dimn_t row_count() const noexcept { return m_stream.size(); }

    scalar_array operator[](dimn_t row) const noexcept;
    scalar operator[](std::pair<dimn_t, dimn_t> index) const noexcept;

    void set_elts_per_row(dimn_t num_elts) noexcept;

    void reserve_size(dimn_t num_rows);

    void push_back(const scalar_pointer& data);
    void push_back(const scalar_array& data);

};

namespace dtl {

template<typename Scalar>
class ESIG_EXPORT intermediate_scalar_interface : public scalar_interface {
public:
    virtual Scalar into_value() const noexcept = 0;
    virtual const Scalar &into_cref() const noexcept = 0;
};

}// namespace dtl

template<typename T>
std::remove_cv_t<std::remove_reference_t<T>>
scalar_cast(const scalar &arg) {
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

namespace dtl {

template<typename Scalar>
class ESIG_EXPORT scalar_type_holder {
    static_assert(!std::is_pointer<Scalar>::value, "Scalar value cannot be a pointer");
    static_assert(!std::is_reference<Scalar>::value, "Scalar value cannot be a reference");

public:
    static const scalar_type *get_type() noexcept;
};

// Explicit implementation for float defined in the library
template<>
ESIG_EXPORT const scalar_type *scalar_type_holder<float>::get_type() noexcept;
//template <> class ESIG_EXPORT scalar_type_holder<float> {
//   public:
//    static const scalar_type *get_type() noexcept;
//};


// Explicit implementation for double defined in the library.
template<>
ESIG_EXPORT const scalar_type *scalar_type_holder<double>::get_type() noexcept;
//template<>
//class ESIG_EXPORT scalar_type_holder<double> {
//public:
//    static const scalar_type *get_type() noexcept;
//};

// explicit instantiation for rational type defined in the library
template<>
ESIG_EXPORT const scalar_type *scalar_type_holder<rational_scalar_type>::get_type() noexcept;




template<typename Scalar>
class scalar_implementation;

template<typename T>
class scalar_type_trait {
    static_assert(!std::is_pointer<T>::value, "scalar cannot be pointer");
    static_assert(!std::is_reference<T>::value, "scalar cannot be reference");
public:
    using value_type = T;
    using rational_type = T;
    using reference = T &;
    using const_reference = const T &;
    using intermediate_interface = intermediate_scalar_interface<T>;
    using rational_intermediate_interface = intermediate_interface;
    using v_wrapper = scalar_implementation<T>;
    using r_wrapper = scalar_implementation<reference>;
    using cr_wrapper = scalar_implementation<const_reference>;

    static const scalar_type *get_type() noexcept {
        return scalar_type_holder<T>::get_type();
    }

    static scalar make(value_type &&arg) {
        return scalar(arg, get_type());
    }
};

template<typename T>
class scalar_type_trait<T &> {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T &;
    using const_reference = const T &;
    using intermediate_interface = intermediate_scalar_interface<T>;

    using rational_intermediate_interface = intermediate_interface;

    using v_wrapper = scalar_implementation<T>;
    using r_wrapper = scalar_implementation<reference>;
    using cr_wrapper = scalar_implementation<const_reference>;

    static const scalar_type *get_type() noexcept {
        return scalar_type_trait<T>::get_type();
    }

    static scalar make(reference arg) {
        return scalar(scalar_pointer(&arg, get_type()));
    }
};

template<typename T>
class scalar_type_trait<const T &> {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T &;
    using const_reference = const T &;
    using intermediate_interface = intermediate_scalar_interface<T>;
    using rational_intermediate_interface = intermediate_interface;
    using v_wrapper = scalar_implementation<T>;
    using r_wrapper = scalar_implementation<reference>;
    using cr_wrapper = scalar_implementation<const_reference>;

    static const scalar_type *get_type() noexcept {
        return scalar_type_trait<T>::get_type();
    }

    static scalar make(const_reference arg) {
        return scalar(scalar_pointer(&arg, get_type()));
    }
};
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

}// namespace dtl

} // namespace scalars
}// namespace esig

#endif//ESIG_COEFFICIENTS_H_
