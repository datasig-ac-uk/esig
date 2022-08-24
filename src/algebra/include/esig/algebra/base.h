//
// Created by user on 23/08/22.
//

#ifndef ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_BASE_H_
#define ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_BASE_H_

#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/algebra_traits.h>
#include <esig/algebra/coefficients.h>
#include <esig/algebra/iteration.h>

#include <memory>

namespace esig {
namespace algebra {


template <typename Algebra>
struct algebra_interface
{
    using algebra_t = Algebra;
    using const_iterator = algebra_iterator;

    virtual ~algebra_interface() = default;

    virtual dimn_t size() const = 0;
    virtual deg_t degree() const = 0;
    virtual deg_t width() const = 0;
    virtual deg_t depth() const = 0;
    virtual vector_type storage_type() const noexcept = 0;
    virtual coefficient_type coeff_type() const noexcept = 0;

    // Element access
    virtual coefficient get(key_type key) const = 0;
    virtual coefficient get_mut(key_type key) = 0;

     // Iteration
    virtual algebra_iterator begin() const = 0;
    virtual algebra_iterator end() const = 0;

    virtual dense_data_access_iterator iterate_dense_components() const = 0;

    // Arithmetic
    virtual Algebra uminus() const = 0;
    virtual Algebra add(const algebra_interface& other) const = 0;
    virtual Algebra sub(const algebra_interface& other) const = 0;
    virtual Algebra smul(const coefficient& other) const = 0;
//    virtual Algebra left_smul(const coefficient& other) const = 0;
//    virtual Algebra right_smul(const coefficient& other) const = 0;
    virtual Algebra sdiv(const coefficient& other) const = 0;
    virtual Algebra mul(const algebra_interface& other) const = 0;

    // Inplace Arithmetic
    virtual void add_inplace(const algebra_interface& other) = 0;
    virtual void sub_inplace(const algebra_interface& other) = 0;
    virtual void smul_inplace(const coefficient& other) = 0;
//    virtual void left_smul_inplace(const coefficient& other) = 0;
//    virtual void right_smul_inplace(const coefficient& other) = 0;
    virtual void sdiv_inplace(const coefficient& other) = 0;
    virtual void mul_inplace(const algebra_interface& other) = 0;

    // Hybrid inplace arithmetic
    virtual void add_scal_mul(const algebra_interface& rhs, const coefficient& coeff);
    virtual void sub_scal_mul(const algebra_interface& rhs, const coefficient& coeff);
    virtual void add_scal_div(const algebra_interface& rhs, const coefficient& coeff);
    virtual void sub_scal_div(const algebra_interface& rhs, const coefficient& coeff);
    virtual void add_mul(const algebra_interface& lhs, const algebra_interface& rhs);
    virtual void sub_mul(const algebra_interface& lhs, const algebra_interface& rhs);
    virtual void mul_smul(const algebra_interface& rhs, const coefficient& coeff);
    virtual void mul_sdiv(const algebra_interface& rhs, const coefficient& coeff);

    // Display methods
    virtual std::ostream& print(std::ostream& os) const = 0;

    // Equality testing
    virtual bool equals(const algebra_interface& other) const = 0;
};

namespace dtl {


template <typename Interface, typename Impl>
class algebra_implementation : public Interface
{
    using context_p = const context*;
    Impl m_data;
    context_p p_ctx;

    static_assert(std::is_base_of<algebra_interface<typename Interface::algebra_t>, Interface>::value,
                  "algebra_interface must be an accessible base of Interface");
public:

    using interface_t = Interface;
    using algebra_t = typename Interface::algebra_t;

    friend class esig::algebra::algebra_implementation_access;

public:
    using scalar_type = typename algebra_info<Impl>::scalar_type;
    using rational_type = typename algebra_info<Impl>::rational_type;

protected:
    static const Impl& cast(const interface_t& arg);
    static Impl& cast(interface_t& arg);


public:

    algebra_implementation(Impl&& impl, context_p ctx);
    algebra_implementation(const Impl& impl, context_p ctx);

    dimn_t size() const override;
    deg_t degree() const override;
    deg_t width() const override;
    deg_t depth() const override;
    vector_type storage_type() const noexcept override;
    coefficient_type coeff_type() const noexcept override;
    coefficient get(key_type key) const override;
    coefficient get_mut(key_type key) override;
    algebra_iterator begin() const override;
    algebra_iterator end() const override;
    dense_data_access_iterator iterate_dense_components() const override;
    algebra_t uminus() const override;
    algebra_t add(const interface_t &other) const override;
    algebra_t sub(const interface_t &other) const override;
    algebra_t smul(const coefficient& scal) const override;
//    algebra_t left_smul(const coefficient &other) const override;
//    algebra_t right_smul(const coefficient &other) const override;
    algebra_t sdiv(const coefficient &other) const override;
    algebra_t mul(const interface_t &other) const override;
    void add_inplace(const interface_t &other) override;
    void sub_inplace(const interface_t &other) override;
    void smul_inplace(const coefficient& other) override;
//    void left_smul_inplace(const coefficient &other) override;
//    void right_smul_inplace(const coefficient &other) override;
    void sdiv_inplace(const coefficient &other) override;
    void mul_inplace(const interface_t &other) override;
    std::ostream &print(std::ostream &os) const override;
    bool equals(const interface_t &other) const override;
};

template <typename Interface, typename Impl>
struct implementation_wrapper_selection
{
    using type = typename std::conditional<std::is_base_of<Interface, Impl>::value, Impl,
        algebra_implementation<Interface, Impl>>::type;
};


template <typename Interface, typename Impl>
using impl_wrapper = typename implementation_wrapper_selection<Interface, Impl>::type;

} // namespace dtl


template <typename Interface>
class algebra_base
{
    explicit algebra_base(std::shared_ptr<Interface> impl);
protected:
    std::shared_ptr<Interface> p_impl;
    friend class algebra_base_access;
    friend class esig::algebra::algebra_implementation_access;

public:
    using interface_t = Interface;

    using algebra_t = typename Interface::algebra_t;
    using const_iterator = typename Interface::const_iterator;

    algebra_base();

    template <typename Impl,
             typename=std::enable_if_t<
                 !std::is_same<
                     std::remove_cv_t<std::remove_reference_t<Impl>>, algebra_t>::value>>
    explicit algebra_base(Impl&& arg, const context* ctx);

    template <typename Impl, typename... Args>
    static algebra_t from_args(Args&&... args);

    const Interface& operator*() const noexcept;
    Interface& operator*() noexcept;

    operator bool() const noexcept;

    dimn_t size() const;
    deg_t width() const;
    deg_t depth() const;
    deg_t degree() const;
    vector_type storage_type() const noexcept;
    coefficient_type coeff_type() const noexcept;

    coefficient operator[](key_type k) const;
    coefficient operator[](key_type k);

    const_iterator begin() const;
    const_iterator end() const;

    dense_data_access_iterator iterate_dense_components() const noexcept;

    // Binary operations
    algebra_t uminus() const;
    algebra_t add(const algebra_t& rhs) const;
    algebra_t sub(const algebra_t& rhs) const;
    algebra_t smul(const coefficient& rhs) const;
//    algebra_t left_smul(const coefficient& scal) const;
//    algebra_t right_smul(const coefficient& scal) const;
    algebra_t sdiv(const coefficient& scal) const;
    algebra_t mul(const algebra_t& rhs) const;

    // In-place binary operations
    algebra_t& add_inplace(const algebra_t& rhs);
    algebra_t& sub_inplace(const algebra_t& rhs);
    algebra_t& smul_inplace(const coefficient& scal);
//    algebra_base& left_smul_inplace(const coefficient& rhs);
//    algebra_base& right_smul_inplace(const coefficient& rhs);
    algebra_t& sdiv_inplace(const coefficient& rhs);
    algebra_t& mul_inplace(const algebra_t& rhs);

    // Hybrid in-place operations
    algebra_t& add_scal_mul(const algebra_t& arg, const coefficient& scal);
    algebra_t& add_scal_div(const algebra_t& arg, const coefficient& scal);
    algebra_t& sub_scal_mul(const algebra_t& arg, const coefficient& scal);
    algebra_t& sub_scal_div(const algebra_t& arg, const coefficient& scal);
    algebra_t& add_mul(const algebra_t& lhs, const algebra_t& rhs);
    algebra_t& sub_mul(const algebra_t& lhs, const algebra_t& rhs);
//    algebra_base& mul_left_smul(const algebra_base& arg, const coefficient& scal);
//    algebra_base& mul_right_smul(const algebra_base& arg, const coefficient& scal);
    algebra_t& mul_smul(const algebra_t& arg, const coefficient& scal);
    algebra_t& mul_sdiv(const algebra_t& arg, const coefficient& scal);


    std::ostream& print(std::ostream& os) const;

    bool operator==(const algebra_t& other) const;
    bool operator!=(const algebra_t& other) const;


};

template<typename Interface>
algebra_base<Interface>::algebra_base() : p_impl(nullptr)
{}

template<typename Interface>
template<typename Impl, typename>
algebra_base<Interface>::algebra_base(Impl &&arg, const context *ctx) :
    p_impl(new dtl::impl_wrapper<Interface, Impl>(std::forward<Impl>(arg), ctx))
{
}
template<typename Interface>
template<typename Impl, typename... Args>
typename algebra_base<Interface>::algebra_t algebra_base<Interface>::from_args(Args &&...args) {
    std::shared_ptr<interface_t> ptr(new dtl::impl_wrapper<Interface, Impl>{std::forward<Args>(args)...});
    return algebra_t(ptr);
}

template<typename Algebra>
void algebra_interface<Algebra>::add_scal_mul(const algebra_interface &rhs, const coefficient &coeff) {
    auto tmp = rhs.smul(coeff);
    this->add_inplace(*tmp);
}
template<typename Algebra>
void algebra_interface<Algebra>::sub_scal_mul(const algebra_interface &rhs, const coefficient &coeff) {
    auto tmp = rhs.smul(coeff);
    this->sub_inplace(*tmp);
}
template<typename Algebra>
void algebra_interface<Algebra>::add_scal_div(const algebra_interface &rhs, const coefficient &coeff) {
    auto tmp = rhs.sdiv(coeff);
    this->add_inplace(*tmp);
}
template<typename Algebra>
void algebra_interface<Algebra>::sub_scal_div(const algebra_interface &rhs, const coefficient &coeff) {
    auto tmp = rhs.sdiv(coeff);
    this->sub_inplace(*tmp);
}
template<typename Algebra>
void algebra_interface<Algebra>::add_mul(const algebra_interface &lhs, const algebra_interface &rhs) {
    auto tmp = lhs.mul(rhs);
    this->add_inplace(*tmp);
}
template<typename Algebra>
void algebra_interface<Algebra>::sub_mul(const algebra_interface &lhs, const algebra_interface &rhs) {
    auto tmp = lhs.mul(rhs);
    this->sub_inplace(*tmp);
}
template<typename Algebra>
void algebra_interface<Algebra>::mul_smul(const algebra_interface &rhs, const coefficient &coeff) {
    auto tmp = rhs.smul(coeff);
    this->mul_inplace(*tmp);
}
template<typename Algebra>
void algebra_interface<Algebra>::mul_sdiv(const algebra_interface &rhs, const coefficient &coeff) {
    auto tmp = rhs.sdiv(coeff);
    this->mul_inplace(*tmp);
}


namespace dtl {
template<typename Interface, typename Impl>
algebra_implementation<Interface, Impl>::algebra_implementation(Impl &&impl, algebra_implementation::context_p ctx)
    : m_data(std::move(impl)), p_ctx(ctx)
{
}
template<typename Interface, typename Impl>
algebra_implementation<Interface, Impl>::algebra_implementation(const Impl &impl, algebra_implementation::context_p ctx)
    : m_data(impl), p_ctx(ctx)
{
}
template<typename Interface, typename Impl>
const Impl &algebra_implementation<Interface, Impl>::cast(const algebra_implementation::interface_t &arg) {
    return dynamic_cast<const algebra_implementation&>(arg).m_data;
}
template<typename Interface, typename Impl>
Impl &algebra_implementation<Interface, Impl>::cast(algebra_implementation::interface_t &arg) {
    return dynamic_cast<algebra_implementation&>(arg).m_data;
}
template<typename Interface, typename Impl>
dimn_t algebra_implementation<Interface, Impl>::size() const {
    return m_data.size();
}
template<typename Interface, typename Impl>
deg_t algebra_implementation<Interface, Impl>::degree() const {
    return m_data.degree();
}
template<typename Interface, typename Impl>
deg_t algebra_implementation<Interface, Impl>::width() const {
    return algebra_info<Impl>::width(m_data);
}
template<typename Interface, typename Impl>
deg_t algebra_implementation<Interface, Impl>::depth() const {
    return algebra_info<Impl>::max_depth(m_data);
}
template<typename Interface, typename Impl>
vector_type algebra_implementation<Interface, Impl>::storage_type() const noexcept {
    return algebra_info<Impl>::vtype();
}
template<typename Interface, typename Impl>
coefficient_type algebra_implementation<Interface, Impl>::coeff_type() const noexcept {
    return algebra_info<Impl>::ctype();
}
template<typename Interface, typename Impl>
coefficient algebra_implementation<Interface, Impl>::get(key_type key) const {
    using info_t = algebra_info<Impl>;
    auto akey = info_t::convert_key(key);
    using ref_t = decltype(m_data[akey]);
    using trait = coefficient_type_trait<ref_t>;
    return trait::make(m_data[akey]);
}
template<typename Interface, typename Impl>
coefficient algebra_implementation<Interface, Impl>::get_mut(key_type key) {
    using info_t = algebra_info<Impl>;
    auto akey = info_t::convert_key(key);
    using ref_t = decltype(m_data[akey]);
    using trait = coefficient_type_trait<ref_t>;
    return trait::make(m_data[akey]);
}
template<typename Interface, typename Impl>
algebra_iterator algebra_implementation<Interface, Impl>::begin() const {
    return algebra_iterator(m_data.begin(), p_ctx);
}
template<typename Interface, typename Impl>
algebra_iterator algebra_implementation<Interface, Impl>::end() const {
    return algebra_iterator(m_data.end(), p_ctx);
}
template<typename Interface, typename Impl>
dense_data_access_iterator algebra_implementation<Interface, Impl>::iterate_dense_components() const {
    return dense_data_access_iterator(
        dtl::dense_data_access_implementation<Impl>(m_data, algebra_info<Impl>::first_key(m_data))
        );
}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::uminus() const {
    return algebra_t(-m_data, p_ctx);
}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::add(const Interface &other) const {
    return algebra_t(m_data + cast(other), p_ctx);
}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::sub(const Interface &other) const {
    return algebra_t(m_data - cast(other), p_ctx);
}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::smul(const coefficient &scal) const {
    return algebra_t(m_data * coefficient_cast<scalar_type>(scal), p_ctx);
}
//template<typename Interface, typename Impl>
//typename Interface::algebra_t algebra_implementation<Interface, Impl>::left_smul(const coefficient &other) const {
//    return algebra_t(m_data*coefficient_cast<scalar_type>(other), p_ctx);
//}
//template<typename Interface, typename Impl>
//typename Interface::algebra_t algebra_implementation<Interface, Impl>::right_smul(const coefficient &other) const {
//    return algebra_t(m_data*coefficient_cast<scalar_type>(other), p_ctx);
//}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::sdiv(const coefficient &other) const {
    return algebra_t(m_data/coefficient_cast<rational_type>(other), p_ctx);
}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::mul(const Interface &other) const {
    return algebra_t(m_data*cast(other), p_ctx);
}
template<typename Interface, typename Impl>
void algebra_implementation<Interface, Impl>::add_inplace(const Interface &other) {
    m_data += cast(other);
}
template<typename Interface, typename Impl>
void algebra_implementation<Interface, Impl>::sub_inplace(const Interface &other) {
    m_data -= cast(other);
}
template<typename Interface, typename Impl>
void algebra_implementation<Interface, Impl>::smul_inplace(const coefficient &other) {
    m_data *= coefficient_cast<scalar_type>(other);
}
//template<typename Interface, typename Impl>
//void algebra_implementation<Interface, Impl>::left_smul_inplace(const coefficient &other) {
//    m_data *= coefficient_cast<scalar_type>(other);
//}
//template<typename Interface, typename Impl>
//void algebra_implementation<Interface, Impl>::right_smul_inplace(const coefficient &other) {
//    m_data *= coefficient_cast<scalar_type>(other);
//}
template<typename Interface, typename Impl>
void algebra_implementation<Interface, Impl>::sdiv_inplace(const coefficient &other) {
    m_data /= coefficient_cast<rational_type>(other);
}
template<typename Interface, typename Impl>
void algebra_implementation<Interface, Impl>::mul_inplace(const Interface &other) {
    m_data *= cast(other);
}
template<typename Interface, typename Impl>
std::ostream &algebra_implementation<Interface, Impl>::print(std::ostream &os) const {
    return os << m_data;
}
template<typename Interface, typename Impl>
bool algebra_implementation<Interface, Impl>::equals(const Interface &other) const {
    try {
        return m_data == cast(other);
    } catch (std::bad_cast&) {
        return false;
    }
}
} // namespace dtl

template<typename Interface>
algebra_base<Interface>::algebra_base(std::shared_ptr<Interface> impl) : p_impl(std::move(impl))
{}
template<typename Interface>
const Interface &algebra_base<Interface>::operator*() const noexcept {
    return *p_impl;
}
template<typename Interface>
Interface &algebra_base<Interface>::operator*() noexcept {
    return *p_impl;
}
template<typename Interface>
algebra_base<Interface>::operator bool() const noexcept {
    return static_cast<bool>(p_impl);
}
template<typename Interface>
dimn_t algebra_base<Interface>::size() const {
    return p_impl->size();
}
template<typename Interface>
deg_t algebra_base<Interface>::width() const {
    return p_impl->width();
}
template<typename Interface>
deg_t algebra_base<Interface>::depth() const {
    return p_impl->depth();
}
template<typename Interface>
deg_t algebra_base<Interface>::degree() const {
    return p_impl->degree();
}
template<typename Interface>
vector_type algebra_base<Interface>::storage_type() const noexcept {
    return p_impl->storage_type();
}
template<typename Interface>
coefficient_type algebra_base<Interface>::coeff_type() const noexcept {
    return p_impl->coeff_type();
}
template<typename Interface>
coefficient algebra_base<Interface>::operator[](key_type k) const {
    return p_impl->get(k);
}
template<typename Interface>
coefficient algebra_base<Interface>::operator[](key_type k) {
    return p_impl->get_mut(k);
}
template<typename Interface>
typename algebra_base<Interface>::const_iterator algebra_base<Interface>::begin() const {
    return p_impl->begin();
}
template<typename Interface>
typename algebra_base<Interface>::const_iterator algebra_base<Interface>::end() const {
    return p_impl->end();
}
template<typename Interface>
dense_data_access_iterator algebra_base<Interface>::iterate_dense_components() const noexcept {
    return p_impl->iterate_dense_components();
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t algebra_base<Interface>::uminus() const {
    return p_impl->uminus();
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t algebra_base<Interface>::add(const algebra_t &rhs) const {
    return p_impl->add(*rhs.p_impl);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t algebra_base<Interface>::sub(const algebra_t &rhs) const {
    return p_impl->sub(*rhs.p_impl);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t algebra_base<Interface>::smul(const coefficient &rhs) const {
    return p_impl->smul(rhs);
}
//template<typename Interface>
//typename algebra_base<Interface>::algebra_t algebra_base<Interface>::left_smul(const coefficient &scal) const {
//    return p_impl->left_smul(scal);
//}
//template<typename Interface>
//typename algebra_base<Interface>::algebra_t algebra_base<Interface>::right_smul(const coefficient &scal) const {
//    return p_impl->right_smul(scal);
//}
template<typename Interface>
typename algebra_base<Interface>::algebra_t algebra_base<Interface>::sdiv(const coefficient &scal) const {
    return p_impl->sdiv(scal);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t algebra_base<Interface>::mul(const algebra_t &rhs) const {
    return p_impl->mul(*rhs.p_impl);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::add_inplace(const algebra_t &rhs) {
    p_impl->add_inplace(*rhs.p_impl);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::sub_inplace(const algebra_t &rhs) {
    p_impl->sub_inplace(*rhs.p_impl);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::smul_inplace(const coefficient &scal) {
    p_impl->smul_inplace(scal);
    return static_cast<algebra_t&>(*this);
}
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
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::sdiv_inplace(const coefficient& rhs) {
    p_impl->sdiv_inplace(rhs);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::mul_inplace(const algebra_t &rhs) {
    p_impl->mul_inplace(*rhs.p_impl);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::add_scal_mul(const algebra_t &arg, const coefficient &scal) {
    p_impl->add_scal_mul(*arg.p_impl, scal);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::add_scal_div(const algebra_t &arg, const coefficient &scal) {
    p_impl->add_scal_div(*arg.p_impl, scal);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::sub_scal_mul(const algebra_t &arg, const coefficient &scal) {
    p_impl->sub_scal_mul(*arg.p_impl, scal);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::sub_scal_div(const algebra_t &arg, const coefficient &scal) {
    p_impl->sub_scal_div(*arg.p_impl, scal);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::add_mul(const algebra_t &lhs, const algebra_t &rhs) {
    p_impl->add_mul(*lhs.p_impl, *rhs.p_impl);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::sub_mul(const algebra_t &lhs, const algebra_t &rhs) {
    p_impl->sub_mul(*lhs.p_impl, *rhs.p_impl);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::mul_smul(const algebra_t &arg, const coefficient &scal) {
    p_impl->mul_smul(*arg.p_impl, scal);
    return static_cast<algebra_t&>(*this);
}
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
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::mul_sdiv(const algebra_t &arg, const coefficient &scal) {
    p_impl->mul_sdiv(*arg.p_impl, scal);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
std::ostream &algebra_base<Interface>::print(std::ostream &os) const {
    return p_impl->print(os);
}
template<typename Interface>
bool algebra_base<Interface>::operator==(const algebra_t &other) const {
    return p_impl->equals(*other.p_impl);
}
template<typename Interface>
bool algebra_base<Interface>::operator!=(const algebra_t &other) const {
    return !p_impl->equals(*other.p_impl);
}

} // namespace algebra
} // namespace esig


#endif//ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_BASE_H_
