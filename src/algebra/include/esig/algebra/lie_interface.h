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

#include <iosfwd>
#include <memory>
#include <type_traits>

namespace esig {
namespace algebra {

class lie;


struct ESIG_ALGEBRA_EXPORT lie_interface {
public:
    using const_iterator = algebra_iterator;

    virtual ~lie_interface() = default;

    // Iteration
    virtual const_iterator begin() const = 0;
    virtual const_iterator end() const = 0;

    // Methods for getting data about the lie
    virtual dimn_t size() const = 0;
    virtual deg_t degree() const = 0;
    virtual deg_t width() const = 0;
    virtual deg_t depth() const = 0;
    virtual vector_type storage_type() const noexcept = 0;
    virtual coefficient_type coeff_type() const noexcept = 0;

    // Element access (via proxy)
    virtual coefficient get(key_type key) const = 0;
    virtual coefficient get_mut(key_type key) = 0;

    // Dense element access
    virtual dense_data_access_iterator iterate_dense_components() const = 0;

    // Arithmetic wrapping
    virtual lie uminus() const = 0;
    virtual lie add(const lie_interface &other) const = 0;
    virtual lie sub(const lie_interface &other) const = 0;
    virtual lie smul(const coefficient &other) const = 0;
    virtual lie sdiv(const coefficient &other) const = 0;
    virtual lie mul(const lie_interface &other) const = 0;

    // Implace arithmetic wrapping
    virtual lie_interface &add_inplace(const lie_interface &other) = 0;
    virtual lie_interface &sub_inplace(const lie_interface &other) = 0;
    virtual lie_interface &smul_inplace(const coefficient &other) = 0;
    virtual lie_interface &sdiv_inplace(const coefficient &other) = 0;
    virtual lie_interface &mul_inplace(const lie_interface &other) = 0;

    // Hybrid inplace arithmetic
    virtual lie_interface &add_scal_mul(const lie_interface &other, const coefficient &scal);
    virtual lie_interface &sub_scal_mul(const lie_interface &other, const coefficient &scal);
    virtual lie_interface &add_scal_div(const lie_interface &other, const coefficient &scal);
    virtual lie_interface &sub_scal_div(const lie_interface &other, const coefficient &scal);
    virtual lie_interface &add_mul(const lie_interface &lhs, const lie_interface &rhs);
    virtual lie_interface &sub_mul(const lie_interface &lhs, const lie_interface &rhs);
    virtual lie_interface &mul_smul(const lie_interface &other, const coefficient &scal);
    virtual lie_interface &mul_sdiv(const lie_interface &other, const coefficient &scal);

    // Display method
    virtual std::ostream &print(std::ostream &os) const = 0;

    // Equality testing
    virtual bool equals(const lie_interface &other) const = 0;
};

struct lie_base_access;


namespace dtl {
/*
 * Typically, the interface will be implemented via the following template wrapper
 * which takes the standard arithmetic operations to implement the arithmetic, and
 * uses traits to identify the properties.
 */

template<typename Impl>
class lie_implementation : public lie_interface
{
    Impl m_data;
    const context *p_ctx;
    friend class ::esig::algebra::lie_base_access;

    using scalar_type = typename algebra_info<Impl>::scalar_type;

public:
    explicit lie_implementation(Impl &&impl, const context *ctx);

    dimn_t size() const override;
    deg_t degree() const override;
    deg_t width() const override;
    deg_t depth() const override;
    vector_type storage_type() const noexcept override;
    coefficient_type coeff_type() const noexcept override;
    const_iterator begin() const override;
    const_iterator end() const override;

    coefficient get(key_type key) const override;
    coefficient get_mut(key_type key) override;

    dense_data_access_iterator iterate_dense_components() const override;

    lie uminus() const override;
    lie add(const lie_interface &other) const override;
    lie sub(const lie_interface &other) const override;
    lie smul(const coefficient &other) const override;
    lie sdiv(const coefficient &other) const override;
    lie mul(const lie_interface &other) const override;
    lie_interface &add_inplace(const lie_interface &other) override;
    lie_interface &sub_inplace(const lie_interface &other) override;
    lie_interface &smul_inplace(const coefficient &other) override;
    lie_interface &sdiv_inplace(const coefficient &other) override;
    lie_interface &mul_inplace(const lie_interface &other) override;
    lie_interface &add_scal_mul(const lie_interface &other, const coefficient &scal) override;
    lie_interface &sub_scal_mul(const lie_interface &other, const coefficient &scal) override;
    lie_interface &add_scal_div(const lie_interface &other, const coefficient &scal) override;
    lie_interface &sub_scal_div(const lie_interface &other, const coefficient &scal) override;
    lie_interface &add_mul(const lie_interface &lhs, const lie_interface &rhs) override;
    lie_interface &sub_mul(const lie_interface &lhs, const lie_interface &rhs) override;
    lie_interface &mul_smul(const lie_interface &other, const coefficient &scal) override;
    lie_interface &mul_sdiv(const lie_interface &other, const coefficient &scal) override;
    std::ostream &print(std::ostream &os) const override;
    bool equals(const lie_interface &other) const override;
};


}// namespace dtl


/**
 * @brief Wrapper class for lie objects
 */
class ESIG_ALGEBRA_EXPORT lie
{
    std::shared_ptr<lie_interface> p_impl;

    friend class algebra_base_access;

    explicit lie(std::shared_ptr<lie_interface> p);
    explicit lie(std::shared_ptr<lie_interface> &&p);

public:
    using interface_t = lie_interface;

    lie() = default;
    lie(const lie &) = default;
    lie(lie &&) noexcept = default;

    template<typename Impl, typename = typename std::enable_if<!std::is_same<std::remove_cv_t<std::remove_reference_t<Impl>>, lie>::value>::type>
    explicit lie(Impl &&impl, const context *ctx);

    template<typename Impl, typename... Args>
    static lie from_args(Args &&...args);

    lie &operator=(const lie &) = default;
    lie &operator=(lie &&) noexcept = default;

    dense_data_access_iterator iterate_dense_components() const;

    lie_interface &operator*() noexcept;
    const lie_interface &operator*() const noexcept;

    deg_t width() const noexcept;
    deg_t depth() const noexcept;
    coefficient_type coeff_type() const noexcept;
    vector_type storage_type() const noexcept;

    dimn_t size() const noexcept;
    deg_t degree() const noexcept;

    // Element access
    coefficient operator[](key_type key) const;
    coefficient operator[](key_type key);

    // Iterators
    algebra_iterator begin() const;
    algebra_iterator end() const;

    // Arithmetic operations
    lie uminus() const;
    lie add(const lie &other) const;
    lie sub(const lie &other) const;
    lie smul(const coefficient &other) const;
    lie sdiv(const coefficient &other) const;
    lie mul(const lie &other) const;

    // Inplace arithmetic
    lie &add_inplace(const lie &other);
    lie &sub_inplace(const lie &other);
    lie &smul_inplace(const coefficient &other);
    lie &sdiv_inplace(const coefficient &other);
    lie &mul_inplace(const lie &other);

    // Fused inplace operations
    lie &add_scal_mul(const lie &other, const coefficient &scal);
    lie &sub_scal_mul(const lie &other, const coefficient &scal);
    lie &add_scal_div(const lie &other, const coefficient &scal);
    lie &sub_scal_div(const lie &other, const coefficient &scal);
    lie &add_mul(const lie &lhs, const lie &rhs);
    lie &sub_mul(const lie &lhs, const lie &rhs);
    lie &mul_smul(const lie &other, const coefficient &scal);
    lie &mul_sdiv(const lie &other, const coefficient &scal);

    std::ostream &print(std::ostream &os) const;

    // Comparison
    bool operator==(const lie &other) const;
    bool operator!=(const lie &other) const;
};


struct lie_base_access {
    template<typename Impl>
    static const Impl &get(const lie &wrapper)
    {
        auto *ptr = algebra_base_access::get(wrapper);
        assert(ptr != nullptr);
        std::cerr << typeid(Impl).name() << '\n'
                  << typeid(*ptr).name() << '\n';
        return dynamic_cast<const dtl::lie_implementation<Impl> &>(*ptr).m_data;
    }

    template<typename Impl>
    static Impl &get(lie &wrapper)
    {
        auto *ptr = algebra_base_access::get(wrapper);
        assert(ptr != nullptr);
        std::cerr << typeid(Impl).name() << '\n'
                  << typeid(*ptr).name() << '\n';
        return dynamic_cast<dtl::lie_implementation<Impl> &>(*ptr).m_data;
    }
};


ESIG_ALGEBRA_EXPORT
std::ostream &operator<<(std::ostream &os, const lie &arg);


template<typename Impl, typename>
lie::lie(Impl &&impl, const context *ctx)
    : p_impl(new dtl::lie_implementation<Impl>(std::forward<Impl>(impl), ctx))
{
}

template<typename Impl, typename... Args>
lie lie::from_args(Args &&...args)
{
    auto ptr = std::shared_ptr<lie_interface>(new Impl(std::forward<Args>(args)...));
    return lie(ptr);
}


// The rest of this file contains the implementation of the template wrapper methods

namespace dtl {
template<typename Impl>
lie_implementation<Impl>::lie_implementation(Impl &&impl, const context *ctx)
    : m_data(std::forward<Impl>(impl)), p_ctx(ctx)
{
}
template<typename Impl>
dimn_t lie_implementation<Impl>::size() const
{
    return m_data.size();
}
template<typename Impl>
deg_t lie_implementation<Impl>::degree() const
{
    return m_data.degree();
}
template<typename Impl>
deg_t lie_implementation<Impl>::width() const
{
    return algebra_info<Impl>::width(m_data);
}
template<typename Impl>
deg_t lie_implementation<Impl>::depth() const
{
    return algebra_info<Impl>::max_depth(m_data);
}
template<typename Impl>
vector_type lie_implementation<Impl>::storage_type() const noexcept
{
    return algebra_info<Impl>::vtype();
}
template<typename Impl>
coefficient_type lie_implementation<Impl>::coeff_type() const noexcept
{
    return algebra_info<Impl>::ctype();
}
template<typename Impl>
lie lie_implementation<Impl>::uminus() const
{
    return lie(-m_data, p_ctx);
}
template<typename Impl>
lie lie_implementation<Impl>::add(const lie_interface &other) const
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    return lie(m_data + o_data, p_ctx);
}
template<typename Impl>
lie lie_implementation<Impl>::sub(const lie_interface &other) const
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    return lie(m_data - o_data, p_ctx);
}
template<typename Impl>
lie lie_implementation<Impl>::smul(const coefficient &other) const
{
    return lie(m_data * coefficient_cast<scalar_type>(other), p_ctx);
}
template<typename Impl>
lie lie_implementation<Impl>::sdiv(const coefficient &other) const
{
    return lie(m_data / coefficient_cast<scalar_type>(other), p_ctx);
}
template<typename Impl>
lie lie_implementation<Impl>::mul(const lie_interface &other) const
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    return lie(m_data * o_data, p_ctx);
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::add_inplace(const lie_interface &other)
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    m_data += o_data;
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::sub_inplace(const lie_interface &other)
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    m_data -= o_data;
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::smul_inplace(const coefficient &other)
{
    m_data *= coefficient_cast<scalar_type>(other);
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::sdiv_inplace(const coefficient &other)
{
    m_data /= coefficient_cast<scalar_type>(other);
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::mul_inplace(const lie_interface &other)
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    m_data *= o_data;
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::add_scal_mul(const lie_interface &other, const coefficient &scal)
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    m_data.add_scal_prod(o_data, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::sub_scal_mul(const lie_interface &other, const coefficient &scal)
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    m_data.sub_scal_prod(o_data, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::add_scal_div(const lie_interface &other, const coefficient &scal)
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    m_data.add_scal_div(o_data, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::sub_scal_div(const lie_interface &other, const coefficient &scal)
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    m_data.sub_scal_div(o_data, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::add_mul(const lie_interface &lhs, const lie_interface &rhs)
{
    auto lhs_data = dynamic_cast<const lie_implementation &>(lhs).m_data;
    auto rhs_data = dynamic_cast<const lie_implementation &>(rhs).m_data;
    m_data.add_mul(lhs_data, rhs_data);
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::sub_mul(const lie_interface &lhs, const lie_interface &rhs)
{
    auto lhs_data = dynamic_cast<const lie_implementation &>(lhs).m_data;
    auto rhs_data = dynamic_cast<const lie_implementation &>(rhs).m_data;
    m_data.sub_mul(lhs_data, rhs_data);
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::mul_smul(const lie_interface &other, const coefficient &scal)
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    m_data.mul_scal_div(o_data, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
lie_interface &lie_implementation<Impl>::mul_sdiv(const lie_interface &other, const coefficient &scal)
{
    auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
    m_data.sub_scal_div(o_data, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
std::ostream &lie_implementation<Impl>::print(std::ostream &os) const
{
    return os << m_data;
}
template<typename Impl>
bool lie_implementation<Impl>::equals(const lie_interface &other) const
{
    try {
        auto o_data = dynamic_cast<const lie_implementation &>(other).m_data;
        return m_data == o_data;
    } catch (std::bad_cast &) {
        return false;
    }
}
template<typename Impl>
lie_interface::const_iterator lie_implementation<Impl>::begin() const
{
    return esig::algebra::lie_interface::const_iterator(m_data.begin(), p_ctx);
}
template<typename Impl>
lie_interface::const_iterator lie_implementation<Impl>::end() const
{
    return esig::algebra::lie_interface::const_iterator(m_data.end(), p_ctx);
}
template<typename Impl>
coefficient lie_implementation<Impl>::get(key_type key) const
{
    using ref_t = decltype(m_data[key]);
    using trait = coefficient_type_trait<ref_t>;
    return trait::make(m_data[key]);
}
template<typename Impl>
coefficient lie_implementation<Impl>::get_mut(key_type key)
{
    using ref_t = decltype(m_data[key]);
    using trait = coefficient_type_trait<ref_t>;
    return trait::make(m_data[key]);
}
template<typename Impl>
dense_data_access_iterator lie_implementation<Impl>::iterate_dense_components() const {
    return dense_data_access_iterator(dtl::dense_data_access_implementation<Impl>(m_data, 1));
}

}// namespace dtl

}// namespace algebra
}// namespace esig


#endif//ESIG_ALGEBRA_LIE_INTERFACE_H_
