//
// Created by user on 02/11/22.
//

#ifndef ESIG_COEFFICIENTS_H_
#define ESIG_COEFFICIENTS_H_

#include "implementation_types.h"
#include "esig_export.h"

#include <cassert>
#include <memory>
#include <string>

namespace esig {

struct scalar_type_info
{
    std::string name;
    int size;
    int alignment;
};


class scalar_type;
class ESIG_EXPORT scalar;
class scalar_pointer;
class scalar_array;



class ESIG_EXPORT scalar_interface
{
public:

    virtual ~scalar_interface() = default;

    virtual const scalar_type* type() const noexcept = 0;

    virtual bool is_const() const noexcept = 0;
    virtual bool is_value() const noexcept = 0;
    virtual bool is_zero() const noexcept = 0;

    virtual scalar_t as_scalar() const = 0;
    virtual void assign(const scalar_interface *other) = 0;

    virtual scalar uminus() const;

    virtual scalar add(const scalar_interface *other) const = 0;
    virtual scalar sub(const scalar_interface *other) const = 0;
    virtual scalar mul(const scalar_interface *other) const = 0;
    virtual scalar div(const scalar_interface *other) const = 0;

    virtual void add_inplace(const scalar_interface *other);
    virtual void sub_inplace(const scalar_interface *other);
    virtual void mul_inplace(const scalar_interface *other);
    virtual void div_inplace(const scalar_interface *other);

    virtual bool equals(const scalar_interface *other) const noexcept;

    virtual std::ostream& print(std::ostream& os) const;
};


class ESIG_EXPORT scalar_type
{
    const scalar_type_info m_info;

    using interface_ptr = std::shared_ptr<scalar_interface>;

public:
    explicit scalar_type(scalar_type_info info) : m_info(info) {}

    // No copy/move constructors or assigments
    scalar_type(const scalar_type&) = delete;
    scalar_type(scalar_type&&) noexcept = delete;
    scalar_type& operator=(const scalar_type&) = delete;
    scalar_type& operator=(scalar_type&&) noexcept = delete;

    virtual ~scalar_type() = default;

    virtual const scalar_type* rational_type() const noexcept;

    const scalar_type_info& info() const noexcept
    { return m_info; }

    virtual interface_ptr from(float) const = 0;
    virtual interface_ptr from(double) const = 0;
    virtual interface_ptr from(int) const = 0;
    virtual interface_ptr from(int, int) const = 0;
    virtual interface_ptr from(long long, long long) const = 0;
    virtual interface_ptr from(const scalar_interface* other) const;

    virtual scalar_pointer allocate(dimn_t) = 0;
    virtual void deallocate(scalar_pointer, dimn_t) = 0;

    virtual interface_ptr construct_at(scalar_pointer, float) = 0;
    virtual interface_ptr construct_at(scalar_pointer, double) = 0;
    virtual interface_ptr construct_at(scalar_pointer, int) = 0;
    virtual interface_ptr construct_at(scalar_pointer, int, int) = 0;
    virtual interface_ptr construct_at(scalar_pointer, long long, long long) = 0;

    virtual interface_ptr one() const;
    virtual interface_ptr mone() const;
    virtual interface_ptr zero() const;

protected:

    /**
     * @brief Create a new scalar pointer instance.
     * @tparam Pointer Type of pointer to be used (e.g. float*)
     * @param val initial value to populate pointer with
     * @return new scalar_pointer instance pointing to val.
     */
    template <typename Pointer>
    scalar_pointer new_scalar_pointer(Pointer val) const;

};

ESIG_EXPORT
inline bool operator==(const scalar_type &lhs, const scalar_type &rhs) noexcept
{ return &lhs == &rhs; }

ESIG_EXPORT
inline bool operator!=(const scalar_type &lhs, const scalar_type &rhs) noexcept
{ return &lhs == &rhs; }

inline std::size_t hash_value(const scalar_type& arg) noexcept
{ return reinterpret_cast<std::size_t>(&arg); }



class scalar
{
    friend class scalar_interface;
    std::shared_ptr<scalar_interface> p_impl;

    scalar_interface* ptr() { return p_impl.get(); }
    const scalar_interface* ptr() const noexcept { return p_impl.get(); }

    explicit scalar(std::shared_ptr<scalar_interface>&& other)
        : p_impl(std::move(other))
    {}

public:

    scalar() = default;
    explicit scalar(const scalar_type*);
    explicit scalar(scalar_t);
    scalar(scalar_t, const scalar_type*);

    scalar(const scalar& other);
    scalar(scalar&& other) noexcept;

    template <typename Scalar>
    explicit scalar(Scalar, const scalar_type* = nullptr)
    {}

    scalar& operator=(const scalar& other);
    scalar& operator=(scalar&& other) noexcept;

    bool is_const() const noexcept
    { return static_cast<bool>(p_impl) || p_impl->is_const(); }
    bool is_value() const noexcept
    { return static_cast<bool>(p_impl) ||  p_impl->is_value(); }
    bool is_zero() const noexcept
    { return static_cast<bool>(p_impl) || p_impl->is_zero(); }

    const scalar_type* type() const noexcept
    { return static_cast<bool>(p_impl) ? nullptr : p_impl->type(); }

    scalar operator-() const;

    scalar operator+(const scalar& other) const;
    scalar operator-(const scalar& other) const;
    scalar operator*(const scalar& other) const;
    scalar operator/(const scalar& other) const;

    scalar& operator+=(const scalar& other);
    scalar& operator-=(const scalar& other);
    scalar& operator*=(const scalar& other);
    scalar& operator/=(const scalar& other);

    bool operator==(const scalar& rhs) const noexcept;
    bool operator!=(const scalar& rhs) const noexcept;

};



class scalar_pointer
{
    void* p_data;
    const scalar_type* p_type;

public:

    scalar operator*() noexcept;
    scalar operator*() const noexcept;

    scalar_pointer operator+(dimn_t index) const noexcept;

    scalar operator[](dimn_t index) noexcept
    { return *(*this + index); }
    scalar operator[](dimn_t index) const noexcept
    { return *(*this + index); }


};


class scalar_array
{
    scalar_pointer p_data;
    dimn_t m_size;

    friend class scalar_type;

    scalar_array(scalar_pointer begin, dimn_t size)
        : p_data(begin), m_size(size)
    {}

public:

    template <typename Int>
    scalar operator[](Int index)
    {
        auto uindex = static_cast<dimn_t>(index);
        assert(0 <= uindex && uindex <= m_size);
        return p_data[uindex];
    }



};


} // namespace esig

#endif//ESIG_COEFFICIENTS_H_
