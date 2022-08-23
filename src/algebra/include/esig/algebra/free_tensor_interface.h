//
// Created by user on 17/03/2022.
//

#ifndef ESIG_ALGEBRA_FREE_TENSOR_INTERFACE_H_
#define ESIG_ALGEBRA_FREE_TENSOR_INTERFACE_H_


#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/coefficients.h>
#include <esig/algebra/iteration.h>

#include <memory>
#include <type_traits>
#include <iosfwd>

namespace esig {
namespace algebra {

class free_tensor;
class context;

struct ESIG_ALGEBRA_EXPORT free_tensor_interface
{
public:

    virtual ~free_tensor_interface() = default;

    // Methods for getting data about the free tensor
    virtual dimn_t size() const = 0;
    virtual deg_t degree() const = 0;
    virtual deg_t width() const = 0;
    virtual deg_t depth() const = 0;
    virtual vector_type storage_type() const noexcept = 0;
    virtual coefficient_type coeff_type() const noexcept = 0;

    virtual coefficient get(key_type key) const = 0;
    virtual coefficient get_mut(key_type key) = 0;

    virtual dense_data_access_iterator iterate_dense_components() const = 0;

    // Iteration
    virtual algebra_iterator begin() const = 0;
    virtual algebra_iterator end() const = 0;

    // Arithmetic wrapping
    virtual free_tensor uminus() const = 0;
    virtual free_tensor add(const free_tensor_interface& other) const = 0;
    virtual free_tensor sub(const free_tensor_interface& other) const = 0;
    virtual free_tensor smul(const coefficient& other) const = 0;
    virtual free_tensor sdiv(const coefficient& other) const = 0;
    virtual free_tensor mul(const free_tensor_interface& other) const = 0;

    // Inplace arithmetic wrapping
    virtual free_tensor_interface& add_inplace(const free_tensor_interface& other) = 0;
    virtual free_tensor_interface& sub_inplace(const free_tensor_interface& other) = 0;
    virtual free_tensor_interface& smul_inplace(const coefficient& other) = 0;
    virtual free_tensor_interface& sdiv_inplace(const coefficient& other) = 0;
    virtual free_tensor_interface& mul_inplace(const free_tensor_interface& other) = 0;

    // Hybrid inplace arithmetic
    virtual free_tensor_interface& add_scal_mul(const free_tensor_interface& other, const coefficient& scal);
    virtual free_tensor_interface& sub_scal_mul(const free_tensor_interface& other, const coefficient& scal);
    virtual free_tensor_interface& add_scal_div(const free_tensor_interface& other, const coefficient& scal);
    virtual free_tensor_interface& sub_scal_div(const free_tensor_interface& other, const coefficient& scal);
    virtual free_tensor_interface& add_mul(const free_tensor_interface& lhs, const free_tensor_interface& rhs);
    virtual free_tensor_interface& sub_mul(const free_tensor_interface& lhs, const free_tensor_interface& rhs);
    virtual free_tensor_interface& mul_smul(const free_tensor_interface& other, const coefficient& scal);
    virtual free_tensor_interface& mul_sdiv(const free_tensor_interface& other, const coefficient& scal);

    // Special functions
    virtual free_tensor exp() const = 0;
    virtual free_tensor log() const = 0;
    virtual free_tensor inverse() const = 0;
    virtual free_tensor_interface& fmexp(const free_tensor_interface& other) = 0;

    // Display methods
    virtual std::ostream& print(std::ostream& os) const = 0;

    // Equality testing
    virtual bool equals(const free_tensor_interface& other) const = 0;
};


struct free_tensor_base_access;


namespace dtl {
/*
 * Typically, the interface will be implemented via the following template wrapper
 * which takes the standard arithmetic operations to implement the arithmetic, and
 * uses traits to identify the properties.
 */

template <typename Impl>
class free_tensor_implementation : public free_tensor_interface
{
    using context_p = const context*;
    Impl m_data;
    context_p p_ctx;
    friend class ::esig::algebra::free_tensor_base_access;
    friend class ::esig::algebra::algebra_implementation_access<free_tensor_implementation, Impl>;

    using scalar_type = typename algebra_info<Impl>::scalar_type;

public:

    explicit free_tensor_implementation(Impl&& impl, context_p ctx);
    explicit free_tensor_implementation(const Impl& impl, context_p ctx);

    dimn_t size() const override;
    deg_t degree() const override;
    deg_t width() const override;
    deg_t depth() const override;
    vector_type storage_type() const noexcept override;
    coefficient_type coeff_type() const noexcept override;
    free_tensor uminus() const override;

    coefficient get(key_type key) const override;
    coefficient get_mut(key_type key) override;

    dense_data_access_iterator iterate_dense_components() const override;

    algebra_iterator begin() const override;
    algebra_iterator end() const override;

    free_tensor add(const free_tensor_interface &other) const override;
    free_tensor sub(const free_tensor_interface &other) const override;
    free_tensor smul(const coefficient &other) const override;
    free_tensor sdiv(const coefficient &other) const override;
    free_tensor mul(const free_tensor_interface &other) const override;
    free_tensor_interface &add_inplace(const free_tensor_interface &other) override;
    free_tensor_interface &sub_inplace(const free_tensor_interface &other) override;
    free_tensor_interface &smul_inplace(const coefficient &other) override;
    free_tensor_interface &sdiv_inplace(const coefficient &other) override;
    free_tensor_interface &mul_inplace(const free_tensor_interface &other) override;
    free_tensor_interface &add_scal_mul(const free_tensor_interface &other, const coefficient &scal) override;
    free_tensor_interface &sub_scal_mul(const free_tensor_interface &other, const coefficient &scal) override;
    free_tensor_interface &add_scal_div(const free_tensor_interface &other, const coefficient &scal) override;
    free_tensor_interface &sub_scal_div(const free_tensor_interface &other, const coefficient &scal) override;
    free_tensor_interface &add_mul(const free_tensor_interface &lhs, const free_tensor_interface &rhs) override;
    free_tensor_interface &sub_mul(const free_tensor_interface &lhs, const free_tensor_interface &rhs) override;
    free_tensor_interface &mul_smul(const free_tensor_interface &other, const coefficient &scal) override;
    free_tensor_interface &mul_sdiv(const free_tensor_interface &other, const coefficient &scal) override;
    free_tensor exp() const override;
    free_tensor log() const override;
    free_tensor inverse() const override;
    free_tensor_interface &fmexp(const free_tensor_interface &other) override;
    bool equals(const free_tensor_interface &other) const override;
    std::ostream &print(std::ostream &os) const override;
};

} // namespace dtl



/**
 * @brief Wrapper class for free tensor objects.
 */
class ESIG_ALGEBRA_EXPORT free_tensor
{
    std::shared_ptr<free_tensor_interface> p_impl;

    friend class algebra_base_access;

    explicit free_tensor(std::shared_ptr<free_tensor_interface> p);
public:
    using interface_t = free_tensor_interface;

    free_tensor() = default;
    free_tensor(const free_tensor &) = default;
    free_tensor(free_tensor &&) noexcept = default;
    free_tensor & operator=(const free_tensor &) = default;
    free_tensor & operator=(free_tensor &&) noexcept = default;

    template <typename Impl, typename = std::enable_if_t<!std::is_same<std::remove_cv_t<std::remove_reference_t<Impl>>, free_tensor>::value>>
    explicit free_tensor(Impl&& impl, const context* ctx);

    template <typename Impl, typename... Args>
    static free_tensor from_args(Args&&... args);

    const free_tensor_interface& operator*() const noexcept;
    free_tensor_interface& operator*() noexcept;

    dense_data_access_iterator iterate_dense_components() const noexcept;

    dimn_t size() const;
    deg_t degree() const;
    deg_t width() const;
    deg_t depth() const;
    vector_type storage_type() const noexcept;
    coefficient_type coeff_type() const noexcept;

    coefficient operator[](key_type key) const;
    coefficient operator[](key_type key);

    algebra_iterator begin() const;
    algebra_iterator end() const;

    free_tensor uminus() const;
    free_tensor add(const free_tensor & other) const;
    free_tensor sub(const free_tensor & other) const;
    free_tensor smul(const coefficient& other) const;
    free_tensor sdiv(const coefficient& other) const;
    free_tensor mul(const free_tensor & other) const;

    free_tensor & add_inplace(const free_tensor & other);
    free_tensor & sub_inplace(const free_tensor & other);
    free_tensor & smul_inplace(const coefficient& other);
    free_tensor & sdiv_inplace(const coefficient& other);
    free_tensor & mul_inplace(const free_tensor & other);

    free_tensor & add_scal_mul(const free_tensor & other, const coefficient& scal);
    free_tensor & sub_scal_mul(const free_tensor & other, const coefficient& scal);
    free_tensor & add_scal_div(const free_tensor & other, const coefficient& scal);
    free_tensor & sub_scal_div(const free_tensor & other, const coefficient& scal);
    free_tensor & add_mul(const free_tensor & lhs, const free_tensor & rhs);
    free_tensor & sub_mul(const free_tensor & lhs, const free_tensor & rhs);
    free_tensor & mul_smul(const free_tensor & other, const coefficient& scal);
    free_tensor & mul_sdiv(const free_tensor & other, const coefficient& scal);

    free_tensor exp() const;
    free_tensor log() const;
    free_tensor inverse() const;
    free_tensor & fmexp(const free_tensor & other);

    // Display method
    std::ostream& print(std::ostream& os) const;

    bool operator==(const free_tensor & other) const;
    bool operator!=(const free_tensor & other) const;
};


struct free_tensor_base_access
{
    template<typename Impl>
    static const Impl &get(const free_tensor &wrapper)
    {
        auto *ptr = algebra_base_access::get(wrapper);
        assert(ptr != nullptr);
        return dynamic_cast<const dtl::free_tensor_implementation<Impl> &>(*ptr).m_data;
    }

    template<typename Impl>
    static Impl &get(free_tensor &wrapper)
    {
        auto *ptr = algebra_base_access::get(wrapper);
        assert(ptr != nullptr);
        return dynamic_cast<dtl::free_tensor_implementation<Impl> &>(*ptr).m_data;
    }
};

ESIG_ALGEBRA_EXPORT
std::ostream& operator<<(std::ostream& os, const free_tensor & arg);




// We have to define the template constructor here too.
template<typename Impl, typename>
free_tensor::free_tensor(Impl &&impl, const context* ctx)
        : p_impl(new dtl::free_tensor_implementation<
        typename std::remove_cv<
                typename std::remove_reference<Impl>::type>::type>(std::forward<Impl>(impl), ctx))
{
}

template<typename Impl, typename... Args>
free_tensor free_tensor::from_args(Args &&...args)
{
    std::shared_ptr<free_tensor_interface> p(new Impl(std::forward<Args>(args)...));
    return free_tensor(p);
}


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
free_tensor_implementation<Impl>::free_tensor_implementation(Impl &&impl, context_p ctx)
    : m_data(std::forward<Impl>(impl)), p_ctx(ctx)
{
}

template<typename Impl>
free_tensor_implementation<Impl>::free_tensor_implementation(const Impl &impl, context_p ctx)
    : m_data(impl), p_ctx(ctx)
{
}

template<typename Impl>
dimn_t free_tensor_implementation<Impl>::size() const
{
    return m_data.size();
}
template<typename Impl>
deg_t free_tensor_implementation<Impl>::degree() const
{
    return m_data.degree();
}
template<typename Impl>
deg_t free_tensor_implementation<Impl>::width() const
{
    return algebra_info<Impl>::width(m_data);
}
template<typename Impl>
deg_t free_tensor_implementation<Impl>::depth() const
{
    return algebra_info<Impl>::max_depth(m_data);
}
template<typename Impl>
vector_type free_tensor_implementation<Impl>::storage_type() const noexcept
{
    return algebra_info<Impl>::vtype();
}
template<typename Impl>
coefficient_type free_tensor_implementation<Impl>::coeff_type() const noexcept
{
    return algebra_info<Impl>::ctype();
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::uminus() const
{
    return free_tensor(-m_data, p_ctx);
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::add(const free_tensor_interface &other) const
{
    return free_tensor(m_data + dynamic_cast<const free_tensor_implementation&>(other).m_data, p_ctx);
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::sub(const free_tensor_interface &other) const
{
    return free_tensor(m_data - dynamic_cast<const free_tensor_implementation&>(other).m_data, p_ctx);
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::smul(const coefficient &other) const
{
    return free_tensor(m_data * coefficient_cast<scalar_type>(other), p_ctx);
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::sdiv(const coefficient &other) const
{
    return free_tensor(m_data / coefficient_cast<scalar_type>(other), p_ctx);
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::mul(const free_tensor_interface &other) const
{
    return free_tensor(m_data * dynamic_cast<const free_tensor_implementation&>(other).m_data, p_ctx);
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::add_inplace(const free_tensor_interface &other)
{
    m_data += dynamic_cast<const free_tensor_implementation&>(other).m_data;
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::sub_inplace(const free_tensor_interface &other)
{
    m_data -= dynamic_cast<const free_tensor_implementation &>(other).m_data;
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::smul_inplace(const coefficient &other)
{
    m_data *= coefficient_cast<scalar_type>(other);
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::sdiv_inplace(const coefficient &other)
{
    m_data /= coefficient_cast<scalar_type>(other);
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::mul_inplace(const free_tensor_interface &other)
{
    m_data *= dynamic_cast<const free_tensor_implementation &>(other).m_data;
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::add_scal_mul(const free_tensor_interface &other, const coefficient &scal)
{
    const auto& dother = dynamic_cast<const free_tensor_implementation&>(other).m_data;
    m_data.add_scal_prod(dother, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::sub_scal_mul(const free_tensor_interface &other, const coefficient &scal)
{
    const auto &dother = dynamic_cast<const free_tensor_implementation &>(other).m_data;
    m_data.sub_scal_prod(dother, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::add_scal_div(const free_tensor_interface &other, const coefficient &scal)
{
    const auto &dother = dynamic_cast<const free_tensor_implementation &>(other).m_data;
    m_data.add_scal_div(dother, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::sub_scal_div(const free_tensor_interface &other, const coefficient &scal)
{
    const auto &dother = dynamic_cast<const free_tensor_implementation &>(other).m_data;
    m_data.sub_scal_div(dother, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::add_mul(const free_tensor_interface &lhs, const free_tensor_interface &rhs)
{
    const auto& dlhs = dynamic_cast<const free_tensor_implementation&>(lhs).m_data;
    const auto& drhs = dynamic_cast<const free_tensor_implementation&>(rhs).m_data;
    m_data.add_mul(dlhs, drhs);
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::sub_mul(const free_tensor_interface &lhs, const free_tensor_interface &rhs)
{
    const auto &dlhs = dynamic_cast<const free_tensor_implementation &>(lhs).m_data;
    const auto &drhs = dynamic_cast<const free_tensor_implementation &>(rhs).m_data;
    m_data.sub_mul(dlhs, drhs);
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::mul_smul(const free_tensor_interface &other, const coefficient &scal)
{
    const auto& dother = dynamic_cast<const free_tensor_implementation&>(other).m_data;
    m_data.mul_scal_prod(dother, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::mul_sdiv(const free_tensor_interface &other, const coefficient &scal)
{
    const auto& dother = dynamic_cast<const free_tensor_implementation&>(other).m_data;
    m_data.mul_scal_div(dother, coefficient_cast<scalar_type>(scal));
    return *this;
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::exp() const
{
    return free_tensor(exp_wrapper(m_data), p_ctx);
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::log() const
{
    return free_tensor(log_wrapper(m_data), p_ctx);
}
template<typename Impl>
free_tensor free_tensor_implementation<Impl>::inverse() const
{
    return free_tensor(inverse_wrapper(m_data), p_ctx);
}
template<typename Impl>
free_tensor_interface &free_tensor_implementation<Impl>::fmexp(const free_tensor_interface &other)
{
    m_data.fmexp_inplace(dynamic_cast<const free_tensor_implementation&>(other).m_data);
    return *this;
}
template<typename Impl>
std::ostream &free_tensor_implementation<Impl>::print(std::ostream &os) const
{
    return os << m_data;
}
template<typename Impl>
bool free_tensor_implementation<Impl>::equals(const free_tensor_interface &other) const
{
    try {
        auto o_data = dynamic_cast<const free_tensor_implementation&>(other).m_data;
        return m_data == o_data;
    } catch (std::bad_cast&) {
        return false;
    }
}
template<typename Impl>
coefficient free_tensor_implementation<Impl>::get(key_type key) const
{
    using info_t = algebra_info<Impl>;
    auto akey = info_t::convert_key(key);
    using ref_t = decltype(m_data[akey]);
    using trait = coefficient_type_trait<ref_t>;
    return trait::make(m_data[akey]);
}
template<typename Impl>
coefficient free_tensor_implementation<Impl>::get_mut(key_type key)
{
    using info_t = algebra_info<Impl>;
    auto akey = info_t::convert_key(key);
    using ref_t = decltype(m_data[akey]);
    using trait = coefficient_type_trait<ref_t>;
    return trait::make(m_data[akey]);
}
template<typename Impl>
algebra_iterator free_tensor_implementation<Impl>::begin() const
{
    return algebra_iterator(m_data.begin(), p_ctx);
}
template<typename Impl>
algebra_iterator free_tensor_implementation<Impl>::end() const
{
    return algebra_iterator(m_data.end());
}
template<typename Impl>
dense_data_access_iterator free_tensor_implementation<Impl>::iterate_dense_components() const {
    return dense_data_access_iterator(
        dtl::dense_data_access_implementation<Impl>(m_data, 0)
            );
}

} // namespace dtl



} // namespace algebra
} // namespace esig



#endif//ESIG_ALGEBRA_FREE_TENSOR_INTERFACE_H_
