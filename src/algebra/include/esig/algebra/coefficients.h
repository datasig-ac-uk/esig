//
// Created by sam on 09/05/22.
//

#ifndef ESIG_COEFFICIENTS_H_
#define ESIG_COEFFICIENTS_H_

#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>


#include <memory>
#include <stdexcept>
#include <type_traits>
#include <iosfwd>



namespace esig {
namespace algebra {


enum class coefficient_type
{
    sp_real,
    dp_real
};

constexpr dimn_t size_of(coefficient_type coeff) noexcept
{
    switch (coeff) {
        case coefficient_type::sp_real:
            return 4;
        case coefficient_type::dp_real:
            return 8;
        default:
            return 0;
    }
}



namespace dtl {

constexpr coefficient_type get_coeff_type(double) noexcept
{
    return coefficient_type::dp_real;
}

constexpr coefficient_type get_coeff_type(float) noexcept
{
    return coefficient_type::sp_real;
}


template<coefficient_type>
struct coeff_type_converter_helper;

}// namespace dtl

template<coefficient_type CType>
using type_of_coeff = typename dtl::coeff_type_converter_helper<CType>::type;





ESIG_ALGEBRA_EXPORT std::ostream& operator<<(std::ostream& os, const coefficient_type& ctype);

class coefficient;


struct ESIG_ALGEBRA_EXPORT coefficient_interface {

    virtual ~coefficient_interface() = default;

    virtual coefficient_type ctype() const noexcept = 0;
    virtual bool is_const() const noexcept = 0;
    virtual bool is_val() const noexcept = 0;

    virtual scalar_t as_scalar() const;
    virtual void assign(coefficient val);

    virtual coefficient add(const coefficient_interface &other) const;
    virtual coefficient sub(const coefficient_interface &other) const;
    virtual coefficient mul(const coefficient_interface &other) const;
    virtual coefficient div(const coefficient_interface &other) const;

    virtual coefficient add(const scalar_t &other) const;
    virtual coefficient sub(const scalar_t &other) const;
    virtual coefficient mul(const scalar_t &other) const;
    virtual coefficient div(const scalar_t &other) const;

    virtual std::ostream& print(std::ostream& os) const;

};


namespace dtl {
template<typename T>
struct underlying_ctype {
    using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};


template<typename T>
class coefficient_implementation
    : public coefficient_interface
{
    T m_data;

    friend struct coefficient_value_helper;

    using coeff_t = typename underlying_ctype<T>::type;

    using coeff_impl = coefficient_implementation<coeff_t>;

    template<typename S>
    friend class coefficient_implementation;

public:
    explicit coefficient_implementation(T &arg);

    coefficient_type ctype() const noexcept override;
    bool is_const() const noexcept override;
    bool is_val() const noexcept override;
    scalar_t as_scalar() const override;

    explicit operator const T&() const noexcept { return m_data; }
    explicit operator T&() noexcept { return m_data; }

    void assign(coefficient other) override;

    coefficient add(const coefficient_interface &other) const override;
    coefficient sub(const coefficient_interface &other) const override;
    coefficient mul(const coefficient_interface &other) const override;
    coefficient div(const coefficient_interface &other) const override;

    std::ostream &print(std::ostream &os) const override;
};



struct coefficient_value_helper {
    template<typename T>
    static const typename std::remove_reference<T>::type &
    value(const coefficient_implementation<T> &proxy)
    {
        return proxy.m_data;
    }

    template<typename T>
    static const typename std::remove_reference<T>::type &
    value(coefficient_implementation<T> &proxy)
    {
        return proxy.m_data;
    }
};


template<typename T>
struct coefficient_type_trait;

}// namespace dtl


class ESIG_ALGEBRA_EXPORT coefficient
{
    std::shared_ptr<coefficient_interface> p_impl;

    template<typename T>
    friend class dtl::coefficient_implementation;

    template<typename T>
    friend struct dtl::coefficient_type_trait;

    explicit coefficient(std::shared_ptr<coefficient_interface> &&arg);

public:
    coefficient();
    explicit coefficient(coefficient_type ctype);
    explicit coefficient(param_t arg);
    coefficient(param_t arg, coefficient_type ctype);
    coefficient(long n, long d, coefficient_type ctype);
    coefficient(long long n, long long d, coefficient_type ctype);

    template<typename C>
    explicit coefficient(C arg);

    explicit operator scalar_t() const noexcept;

    bool is_const() const noexcept;
    bool is_value() const noexcept;
    coefficient_type ctype() const noexcept;

    const coefficient_interface& operator*() const noexcept;


    coefficient &operator=(const coefficient &other);

    coefficient operator-() const;

    coefficient &operator+=(const coefficient &other);
    coefficient &operator-=(const coefficient &other);
    coefficient &operator*=(const coefficient &other);
    coefficient &operator/=(const coefficient &other);

    coefficient operator+(const coefficient &other) const;
    coefficient operator-(const coefficient &other) const;
    coefficient operator*(const coefficient &other) const;
    coefficient operator/(const coefficient &other) const;

    coefficient &operator+=(const scalar_t &other);
    coefficient &operator-=(const scalar_t &other);
    coefficient &operator*=(const scalar_t &other);
    coefficient &operator/=(const scalar_t &other);
};

template<typename T>
T coefficient_cast(coefficient s)
{
    const auto &s_ref = *s;
    if (typeid(s_ref) == typeid(dtl::coefficient_implementation<T>)) {
        const auto &t_ref = dynamic_cast<const dtl::coefficient_implementation<T>&>(s_ref);
        return T(static_cast<const T&>(t_ref));
    }

    return T(static_cast<scalar_t>(s));
}



struct ESIG_ALGEBRA_EXPORT data_allocator
{
    using value_type = char;
    using pointer = char*;
    using const_pointer = const char*;
    using reference = char&;
    using const_reference = const char&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    virtual ~data_allocator() = default;

    virtual char* allocate(size_type n) = 0;
    virtual void deallocate(char* p, size_type n) = 0;

    virtual size_type item_size() const noexcept = 0;
    virtual size_type alignment() const noexcept = 0;
};


ESIG_ALGEBRA_EXPORT
std::shared_ptr<data_allocator> allocator_for_coeff(coefficient_type ctype);

ESIG_ALGEBRA_EXPORT
std::shared_ptr<data_allocator> allocator_for_key_coeff(coefficient_type ctype);

/**
 * Information about a particular coefficient
 */


class ESIG_ALGEBRA_EXPORT data_buffer
{
protected:
    char* data_begin = nullptr;
    char* data_end = nullptr;
    bool is_const = false;
public:
    using size_type = dimn_t;
    using pointer = char*;
    using const_pointer = const char*;

    data_buffer(char* begin, char* end, bool constant);
    data_buffer(const char*, const char*);

    void swap(data_buffer& other) noexcept;

    pointer begin() noexcept;
    pointer end() noexcept;
    const_pointer begin() const noexcept;
    const_pointer end() const noexcept;
    size_type size() const noexcept;
};


class ESIG_ALGEBRA_EXPORT allocating_data_buffer : public data_buffer
{
protected:
    using allocator_type = std::shared_ptr<data_allocator>;
    allocator_type m_alloc;
    typename data_buffer::size_type m_size;
public:

    using typename data_buffer::size_type;
    using typename data_buffer::pointer;
    using typename data_buffer::const_pointer;

    allocating_data_buffer()
        : data_buffer(nullptr, nullptr, false), m_alloc(nullptr), m_size(0)
    {}
    void swap(allocating_data_buffer& other) noexcept;

    void set_allocator_and_alloc(std::shared_ptr<data_allocator>&& alloc, dimn_t size);

    allocating_data_buffer(const allocating_data_buffer& other);
    allocating_data_buffer(allocating_data_buffer&& other) noexcept;

    allocating_data_buffer(allocator_type alloc, size_type n);
    allocating_data_buffer(allocator_type alloc, const char* begin, const char* end);
    ~allocating_data_buffer();

    allocating_data_buffer& operator=(const allocating_data_buffer& other);
    allocating_data_buffer& operator=(allocating_data_buffer&& other) noexcept;

    typename data_buffer::size_type item_size() const noexcept;
};


class ESIG_ALGEBRA_EXPORT rowed_data_buffer : public allocating_data_buffer
{
    size_type row_size = 0;
    using allocator_type = typename allocating_data_buffer::allocator_type;

public:
    using range_type = std::pair<const char *, const char *>;
    rowed_data_buffer();
    rowed_data_buffer(allocator_type alloc, size_type row_size, size_type nrows);
    rowed_data_buffer(allocator_type alloc, size_type row_size, const char *, const char *);

    range_type operator[](size_type rowid) const noexcept;

};


namespace dtl {

template<typename BaseAlloc>
class allocator_ext : public data_allocator, BaseAlloc// EBC optimisation
{
    using traits = std::allocator_traits<BaseAlloc>;

public:
    char *allocate(size_type n) override;
    void deallocate(char *p, size_type n) override;
    size_type item_size() const noexcept override;
    size_type alignment() const noexcept override;
};
template<typename BaseAlloc>
char *allocator_ext<BaseAlloc>::allocate(data_allocator::size_type n)
{
    return reinterpret_cast<char *>(traits::allocate(*this, n));
}
template<typename BaseAlloc>
void allocator_ext<BaseAlloc>::deallocate(char *p, data_allocator::size_type n)
{
    traits::deallocate(*this, reinterpret_cast<typename traits::pointer>(p), n);
}
template<typename BaseAlloc>
data_allocator::size_type allocator_ext<BaseAlloc>::item_size() const noexcept
{
    return sizeof(typename traits::value_type);
}
template<typename BaseAlloc>
data_allocator::size_type allocator_ext<BaseAlloc>::alignment() const noexcept
{
    return alignof(typename traits::value_type);
}


}// namespace dtl


template<typename C>
coefficient::coefficient(C arg)
        : p_impl(new dtl::coefficient_implementation<C>(arg))
{}
namespace dtl {

template<typename T>
struct coefficient_type_trait {

    using value_type = T;
    using reference = T &;
    using const_reference = const T &;

    using value_wrapper = coefficient_implementation<T>;
    using reference_wrapper = coefficient_implementation<reference>;
    using const_reference_wrapper = coefficient_implementation<const_reference>;

    static coefficient make(T arg)
    {
        return coefficient(
                std::shared_ptr<coefficient_interface>(new value_wrapper{arg}));
    }
};

template<typename T>
struct coefficient_type_trait<T &> {
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;

    using value_wrapper = coefficient_implementation<T>;
    using reference_wrapper = coefficient_implementation<reference>;
    using const_reference_wrapper = coefficient_implementation<const_reference>;

    static coefficient make(reference arg)
    {
        return coefficient(
                std::shared_ptr<coefficient_interface>(new reference_wrapper(arg)));
    }
};

template<typename T>
struct coefficient_type_trait<const T &> {
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;

    using value_wrapper = coefficient_implementation<T>;
    using reference_wrapper = coefficient_implementation<reference>;
    using const_reference_wrapper = coefficient_implementation<const_reference>;

    static coefficient make(const_reference arg)
    {
        return coefficient(
                std::shared_ptr<coefficient_interface>(new const_reference_wrapper(arg)));
    }
};

template<typename T>
coefficient_implementation<T>::coefficient_implementation(T &arg)
    : m_data(arg)
{}

template<typename T>
coefficient_type coefficient_implementation<T>::ctype() const noexcept
{
    return get_coeff_type(coeff_t(0));
}

template<typename T>
bool coefficient_implementation<T>::is_const() const noexcept
{
    return std::is_const<T>::value;
}
template<typename T>
bool coefficient_implementation<T>::is_val() const noexcept
{
    return std::is_same<T, coeff_t>::value;
}
template<typename T>
scalar_t coefficient_implementation<T>::as_scalar() const
{
    return static_cast<scalar_t>(m_data);
}

template<typename T>
void assign_helper(T &arg, const T &other)
{
    arg = other;
}
template<typename T>
void assign_helper(const T &arg, const T &other)
{
    throw std::runtime_error("cannot assign to const value");
}

template<typename T>
void coefficient_implementation<T>::assign(coefficient other)
{
    auto &tid = typeid(*other.p_impl);
    using trait = coefficient_type_trait<T>;
    if (tid == typeid(typename trait::value_wrapper &)) {
        auto o_data = dynamic_cast<typename trait::value_wrapper &>(*other.p_impl).m_data;
        assign_helper(m_data, o_data);
    } else if (tid == typeid(typename trait::reference_wrapper &)) {
        auto o_data = dynamic_cast<typename trait::reference_wrapper &>(*other.p_impl).m_data;
        assign_helper(m_data, o_data);
    } else if (tid == typeid(typename trait::const_reference_wrapper &)) {
        auto o_data = dynamic_cast<typename trait::const_reference_wrapper &>(*other.p_impl).m_data;
        assign_helper(m_data, o_data);
    } else {
        throw std::bad_cast();
    }
}

#define ESIG_ALGEBRA_GENERATE_DEFN(NAME, OP)                                                           \
    template<typename T>                                                                               \
    coefficient                                                                                  \
    coefficient_implementation<T>::NAME(const coefficient_interface &other) const                      \
    {                                                                                                  \
        auto &tid = typeid(other);                                                                     \
        using trait = coefficient_type_trait<T>;                                                       \
        if (tid == typeid(const typename trait::value_wrapper &)) {                                    \
            auto o_data = dynamic_cast<const typename trait::value_wrapper &>(other).m_data;           \
            return coefficient(m_data OP o_data);                                                \
        } else if (tid == typeid(const typename trait::reference_wrapper &)) {                         \
            auto o_data = dynamic_cast<const typename trait::reference_wrapper &>(other).m_data;       \
            return coefficient(m_data OP o_data);                                                \
        } else if (tid == typeid(const typename trait::const_reference_wrapper &)) {                   \
            auto o_data = dynamic_cast<const typename trait::const_reference_wrapper &>(other).m_data; \
            return coefficient(m_data OP o_data);                                                \
        } else {                                                                                       \
            throw std::bad_cast();                                                                     \
        }                                                                                              \
    }

ESIG_ALGEBRA_GENERATE_DEFN(add, +)
ESIG_ALGEBRA_GENERATE_DEFN(sub, -)
ESIG_ALGEBRA_GENERATE_DEFN(mul, *)
ESIG_ALGEBRA_GENERATE_DEFN(div, /)

template<typename T>
std::ostream &coefficient_implementation<T>::print(std::ostream &os) const
{
    return os << m_data;
}

#undef ESIG_ALGEBRA_GENERATE_DEFN


template<>
struct coeff_type_converter_helper<coefficient_type::dp_real> {
    using type = double;
};

template<>
struct coeff_type_converter_helper<coefficient_type::sp_real> {
    using type = float;
};




}// namespace dtl


}// namespace algebra
}// namespace esig

#endif//ESIG_COEFFICIENTS_H_
