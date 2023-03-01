//
// Created by user on 23/08/22.
//

#ifndef ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_BASE_H_
#define ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_BASE_H_

#include <esig/implementation_types.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <iosfwd>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/optional.hpp>
#include <boost/type_traits/copy_cv.hpp>
#include <boost/type_traits/copy_cv_ref.hpp>
#include <boost/type_traits/is_detected.hpp>

#include <esig/scalars.h>
#include "esig/algebra/esig_algebra_export.h"
#include "algebra_traits.h"
#include "iteration.h"


namespace esig {
namespace algebra {

enum class ImplementationType {
    borrowed,
    owned
};

enum class VectorType {
    sparse,
    dense
};

template <typename Algebra>
struct algebra_info;


class ESIG_ALGEBRA_EXPORT BasisInterface {
public:

    virtual ~BasisInterface() = default;

    virtual boost::optional<deg_t> width() const noexcept;
    virtual boost::optional<deg_t> depth() const noexcept;
    virtual boost::optional<deg_t> degree(key_type key) const noexcept;
    virtual std::string key_to_string(key_type key) const noexcept;
    virtual dimn_t size(int) const noexcept;
    virtual dimn_t start_of_degree(int) const noexcept;
    virtual boost::optional<key_type> lparent(key_type key) const noexcept;
    virtual boost::optional<key_type> rparent(key_type key) const noexcept;
    virtual key_type index_to_key(dimn_t idx) const noexcept;
    virtual dimn_t key_to_index(key_type key) const noexcept;
    virtual let_t first_letter(key_type key) const noexcept;
    virtual key_type key_of_letter(let_t letter) const noexcept;
    virtual bool letter(key_type key) const noexcept;
};


namespace dtl {

template <typename Impl>
class BasisImplementation : public BasisInterface {
    const Impl* p_basis;
    using traits = basis_info<Impl>;

public:
    BasisImplementation(const Impl* arg) : p_basis(arg)
    {}

    boost::optional<deg_t> width() const noexcept override {
        return p_basis->width();
    }
    boost::optional<deg_t> depth() const noexcept override {
        return p_basis->depth();
    }
    boost::optional<deg_t> degree(key_type key) const noexcept override {
        return traits::degree(*p_basis, key);
    }
    std::string key_to_string(key_type key) const noexcept override {
        std::stringstream ss;
        p_basis->print_key(ss, traits::convert_key(*p_basis, key));
        return ss.str();
    }
    dimn_t size(int i) const noexcept override {
        return p_basis->size(i);
    }
    dimn_t start_of_degree(int i) const noexcept override {
        return p_basis->start_of_degree(i);
    }
    boost::optional<key_type> lparent(key_type key) const noexcept override {
        return traits::convert_key(*p_basis, p_basis->lparent(traits::convert_key(*p_basis, key)));
    }
    boost::optional<key_type> rparent(key_type key) const noexcept override {
        return traits::convert_key(*p_basis, p_basis->rparent(traits::convert_key(*p_basis, key)));
    }
    key_type index_to_key(dimn_t idx) const noexcept override {
        return traits::convert_key(*p_basis, p_basis->index_to_key(idx));
    }
    dimn_t key_to_index(key_type key) const noexcept override {
        return dimn_t(p_basis->key_to_index(traits::convert_key(*p_basis, key)));
    }
    let_t first_letter(key_type key) const noexcept override {
        return p_basis->first_letter(traits::convert_key(*p_basis, key));
    }
    key_type key_of_letter(let_t letter) const noexcept override {
        return traits::convert_key(*p_basis, p_basis->key_of_letter(letter));
    }
    bool letter(key_type key) const noexcept override {
        return p_basis->letter(traits::convert_key(*p_basis, key));
    }
};

} // namespace dtl


class ESIG_ALGEBRA_EXPORT Basis {
    std::unique_ptr<const BasisInterface> p_impl;

public:

    template <typename T>
    Basis(const T* arg) : p_impl(new dtl::BasisImplementation<T>(arg))
    {}

    const BasisInterface & operator*() const noexcept
    {
        return *p_impl;
    }

    boost::optional<deg_t> width() const noexcept;
    boost::optional<deg_t> depth() const noexcept;
    boost::optional<deg_t> degree(key_type key) const noexcept;
    std::string key_to_string(key_type key) const noexcept;
    dimn_t size(int deg) const noexcept;
    dimn_t start_of_degree(int deg) const noexcept;
    boost::optional<key_type> lparent(key_type key) const noexcept;
    boost::optional<key_type> rparent(key_type key) const noexcept;
    key_type index_to_key(dimn_t idx) const noexcept;
    dimn_t key_to_index(key_type key) const noexcept;
    let_t first_letter(key_type) const noexcept;
    key_type key_of_letter(let_t letter) const noexcept;
    bool letter(key_type key) const noexcept;

};



template <typename Algebra>
struct AlgebraInterface {
    using algebra_t = Algebra;
    using const_iterator = algebra_iterator;

    virtual ~AlgebraInterface() = default;

    virtual dimn_t size() const = 0;
    virtual deg_t degree() const;
    virtual deg_t width() const;
    virtual deg_t depth() const;
    virtual VectorType storage_type() const noexcept = 0;
    virtual const esig::scalars::ScalarType * coeff_type() const noexcept = 0;

    // Element access
    virtual scalars::Scalar get(key_type key) const = 0;
    virtual scalars::Scalar get_mut(key_type key) = 0;

     // Iteration
    virtual algebra_iterator begin() const = 0;
    virtual algebra_iterator end() const = 0;

    virtual scalars::ScalarArray dense_data() const;
//    virtual dense_data_access_iterator iterate_dense_components() const = 0;

    // Arithmetic
    virtual Algebra uminus() const = 0;
    virtual Algebra add(const AlgebraInterface & other) const = 0;
    virtual Algebra sub(const AlgebraInterface & other) const = 0;
    virtual Algebra smul(const scalars::Scalar & other) const = 0;
//    virtual Algebra left_smul(const coefficient& other) const = 0;
//    virtual Algebra right_smul(const coefficient& other) const = 0;
    virtual Algebra sdiv(const scalars::Scalar & other) const = 0;
    virtual Algebra mul(const AlgebraInterface & other) const = 0;

    // Inplace Arithmetic
    virtual void add_inplace(const AlgebraInterface & other) = 0;
    virtual void sub_inplace(const AlgebraInterface & other) = 0;
    virtual void smul_inplace(const scalars::Scalar & other) = 0;
//    virtual void left_smul_inplace(const coefficient& other) = 0;
//    virtual void right_smul_inplace(const coefficient& other) = 0;
    virtual void sdiv_inplace(const scalars::Scalar & other) = 0;
    virtual void mul_inplace(const AlgebraInterface & other) = 0;

    // Hybrid inplace arithmetic
    virtual void add_scal_mul(const AlgebraInterface & rhs, const scalars::Scalar & coeff);
    virtual void sub_scal_mul(const AlgebraInterface & rhs, const scalars::Scalar & coeff);
    virtual void add_scal_div(const AlgebraInterface & rhs, const scalars::Scalar & coeff);
    virtual void sub_scal_div(const AlgebraInterface & rhs, const scalars::Scalar & coeff);
    virtual void add_mul(const AlgebraInterface & lhs, const AlgebraInterface & rhs);
    virtual void sub_mul(const AlgebraInterface & lhs, const AlgebraInterface & rhs);
    virtual void mul_smul(const AlgebraInterface & rhs, const scalars::Scalar & coeff);
    virtual void mul_sdiv(const AlgebraInterface & rhs, const scalars::Scalar & coeff);

    // Display methods
    virtual std::ostream& print(std::ostream& os) const = 0;

    // Equality testing
    virtual bool equals(const AlgebraInterface & other) const = 0;


    virtual std::uintptr_t id() const noexcept;
    virtual ImplementationType type() const noexcept;

};


template <typename Base, typename Fibre>
struct AlgebraBundleInterface : public Base
{
    virtual Fibre fibre() = 0;
};

template <typename Argument, typename Result=Argument>
struct OperatorInterface {
    using argument_type = Argument;
    using result_type = Result;

    virtual ~OperatorInterface() = default;

    virtual result_type apply(const argument_type& argument) const = 0;
};


template <typename Interface>
struct algebra_access;


namespace dtl {

template <typename Interface, typename Impl, typename Fn>
Impl fallback_binary_op(const Impl& lhs, const Interface& rhs, Fn op);

template <typename Interface, typename Impl, typename Fn>
void fallback_inplace_binary_op(Impl& lhs, const Interface& rhs, Fn op);

template <typename Interface, typename Impl, typename Fn>
void fallback_multiplication(Impl& result, const Interface& lhs, const Interface& rhs, Fn Op);

template <typename Interface, typename Impl, typename Fn>
void fallback_multiplication(Impl& result, const Impl& lhs, const Interface& rhs, Fn op);

template <typename Interface, typename Impl, typename Fn>
void fallback_multiplication(Impl& result, const Interface& lhs, const Impl& rhs, Fn op);

template <typename Interface, typename Impl, typename Fn>
void fallback_inplace_multiplication(Impl& result, const Interface& rhs, Fn Op);

template<typename It>
struct multiplication_data {
    It begin;
    It end;
    std::vector<It> degree_ranges;
    deg_t degree;
};

template<typename Algebra, typename LhsIt, typename RhsIt, typename Op>
void fallback_multiplication_impl(Algebra &result,
                             const multiplication_data<LhsIt>& lhs,
                             const multiplication_data<RhsIt>& rhs,
                             Op op);

template <typename Algebra, typename LhsIt, typename RhsIt>
struct multiplication_dispatcher
{
    template <typename Op>
    static void dispatch(Algebra& result,
                         const multiplication_data<LhsIt>& lhs,
                         const multiplication_data<RhsIt>& rhs,
                         Op op)
    {
        fallback_multiplication_impl(result, lhs, rhs, op);
    }
};
template<typename Algebra, typename LhsIt, typename RhsIt, typename Op>
void multiply_and_add(Algebra &result,
                      dtl::multiplication_data<LhsIt> lhs,
                      dtl::multiplication_data<RhsIt> rhs,
                      Op op
                      )
{
    /*
     * The dispatcher can be specialised by specific implementations to
     * use a native multiplication operation where possible. Otherwise it
     * uses the fallback multiplication defined in algebra_traits.h
     */
    dtl::multiplication_dispatcher<Algebra, LhsIt, RhsIt>::dispatch(result, lhs, rhs, op);
}


template <typename Interface, typename Impl>
class borrowed_algebra_implementation;

template <typename Interface, typename Impl>
class algebra_implementation : public Interface
{
protected:
    using context_p = const context*;
    Impl m_data;
    context_p p_ctx;


    static_assert(std::is_base_of<AlgebraInterface<typename Interface::algebra_t>, Interface>::value,
                  "algebra_interface must be an accessible base of Interface");
public:

    using interface_t = Interface;
    using algebra_t = typename Interface::algebra_t;
    using algebra_interface_t = AlgebraInterface<algebra_t>;

    friend class ::esig::algebra::algebra_access<Interface>;
    friend class ::esig::algebra::algebra_access<algebra_interface_t>;
    friend class borrowed_algebra_implementation<Interface, Impl>;
    using borrowed_alg_impl_t = borrowed_algebra_implementation<Interface, Impl>;

public:
    using this_scalar_type = typename algebra_info<Impl>::scalar_type;
    using rational_type = typename algebra_info<Impl>::rational_type;

protected:
    template <typename T>
    static T cast(const algebra_interface_t& arg) noexcept
    {
        if (arg.type() == ImplementationType::owned) {
            return static_cast<boost::copy_cv_t<algebra_implementation, std::remove_reference_t<T>>&>(arg).m_data;
        } else {
            return *static_cast<boost::copy_cv_t<borrowed_alg_impl_t, std::remove_reference_t<T>>&>(arg).p_impl;
        }
    }

public:
    std::uintptr_t id() const noexcept override;


    algebra_implementation(Impl&& impl, context_p ctx);
    algebra_implementation(const Impl& impl, context_p ctx);

    dimn_t size() const override;
    deg_t degree() const override;
    deg_t width() const override;
    deg_t depth() const override;
    VectorType storage_type() const noexcept override;
    const esig::scalars::ScalarType *coeff_type() const noexcept override;
    scalars::Scalar get(key_type key) const override;
    scalars::Scalar get_mut(key_type key) override;
    algebra_iterator begin() const override;
    algebra_iterator end() const override;

private:

    template <typename T>
    using has_as_ptr_t = decltype(std::declval<const T&>().as_ptr());

    template <typename B, typename=std::enable_if_t<boost::is_detected<has_as_ptr_t, B>::value>>
    scalars::ScalarArray dense_data_impl(const B& data) const {
        return {data.as_ptr(), coeff_type(), data.dimension()};
    }

    scalars::ScalarArray dense_data_impl(...) const {
        return Interface::dense_data();
    }

public:

    scalars::ScalarArray dense_data() const override {
        return dense_data_impl(m_data.base_vector());
    }
//    dense_data_access_iterator iterate_dense_components() const override;
    algebra_t uminus() const override;
    algebra_t add(const algebra_interface_t &other) const override;
    algebra_t sub(const algebra_interface_t &other) const override;
    algebra_t smul(const scalars::Scalar & scal) const override;
//    algebra_t left_smul(const coefficient &other) const override;
//    algebra_t right_smul(const coefficient &other) const override;
    algebra_t sdiv(const scalars::Scalar &other) const override;
    algebra_t mul(const algebra_interface_t &other) const override;
    void add_inplace(const algebra_interface_t &other) override;
    void sub_inplace(const algebra_interface_t &other) override;
    void smul_inplace(const scalars::Scalar & other) override;
//    void left_smul_inplace(const coefficient &other) override;
//    void right_smul_inplace(const coefficient &other) override;
    void sdiv_inplace(const scalars::Scalar &other) override;
    void mul_inplace(const algebra_interface_t &other) override;
    std::ostream &print(std::ostream &os) const override;
    bool equals(const algebra_interface_t &other) const override;
};

template <typename Interface, typename Impl>
class borrowed_algebra_implementation : public Interface
{
    using context_p = const context*;

    Impl* p_impl;
    context_p p_ctx;
    using impl_base_type = std::remove_cv_t<Impl>;

    using impl_type = Impl;

public:
    using interface_t = Interface;
    using algebra_t = typename Interface::algebra_t;
    using algebra_interface_t = AlgebraInterface<algebra_t>;

private:
    friend class ::esig::algebra::algebra_access<Interface>;
    friend class ::esig::algebra::algebra_access<algebra_interface_t>;
    friend class algebra_implementation<Interface, Impl>;
    using alg_impl_t = algebra_implementation<Interface, Impl>;

public:

    std::uintptr_t id() const noexcept override;
    ImplementationType type() const noexcept override;


    using this_scalar_type = typename algebra_info<impl_base_type>::scalar_type;
    using rational_type = typename algebra_info<impl_base_type>::rational_type;

    borrowed_algebra_implementation(Impl* impl, context_p ctx);

    dimn_t size() const override;
    deg_t degree() const override;
    deg_t width() const override;
    deg_t depth() const override;
    VectorType storage_type() const noexcept override;
    const esig::scalars::ScalarType *coeff_type() const noexcept override;
    scalars::Scalar get(key_type key) const override;
    scalars::Scalar get_mut(key_type key) override;
    algebra_iterator begin() const override;
    algebra_iterator end() const override;
//    dense_data_access_iterator iterate_dense_components() const override;
    algebra_t uminus() const override;
    algebra_t add(const algebra_interface_t &other) const override;
    algebra_t sub(const algebra_interface_t &other) const override;
    algebra_t smul(const scalars::Scalar & scal) const override;
//    algebra_t left_smul(const coefficient &other) const override;
//    algebra_t right_smul(const coefficient &other) const override;
    algebra_t sdiv(const scalars::Scalar &other) const override;
    algebra_t mul(const algebra_interface_t &other) const override;
    void add_inplace(const algebra_interface_t &other) override;
    void sub_inplace(const algebra_interface_t &other) override;
    void smul_inplace(const scalars::Scalar & other) override;
//    void left_smul_inplace(const coefficient &other) override;
//    void right_smul_inplace(const coefficient &other) override;
    void sdiv_inplace(const scalars::Scalar &other) override;
    void mul_inplace(const algebra_interface_t &other) override;
    std::ostream &print(std::ostream &os) const override;
    bool equals(const algebra_interface_t &other) const override;

};



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
class algebra_base
{
    explicit algebra_base(std::shared_ptr<Interface> impl);
protected:
    std::shared_ptr<Interface> p_impl;
    friend class algebra_base_access;

    friend class algebra_access<Interface>;
    friend class algebra_access<AlgebraInterface<typename Interface::algebra_t>>;

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

    operator bool() const noexcept; // NOLINT(google-explicit-constructor)

    dimn_t size() const;
    deg_t width() const;
    deg_t depth() const;
    deg_t degree() const;
    VectorType storage_type() const noexcept;
    const scalars::ScalarType * coeff_type() const noexcept;

    scalars::Scalar operator[](key_type k) const;
    scalars::Scalar operator[](key_type k);

    const_iterator begin() const;
    const_iterator end() const;

    scalars::ScalarArray dense_data() const { return p_impl->dense_data(); }
//    dense_data_access_iterator iterate_dense_components() const noexcept;

    // Binary operations
    algebra_t uminus() const;
    algebra_t add(const algebra_t& rhs) const;
    algebra_t sub(const algebra_t& rhs) const;
    algebra_t smul(const scalars::Scalar & rhs) const;
//    algebra_t left_smul(const coefficient& scal) const;
//    algebra_t right_smul(const coefficient& scal) const;
    algebra_t sdiv(const scalars::Scalar & scal) const;
    algebra_t mul(const algebra_t& rhs) const;

    // In-place binary operations
    algebra_t& add_inplace(const algebra_t& rhs);
    algebra_t& sub_inplace(const algebra_t& rhs);
    algebra_t& smul_inplace(const scalars::Scalar & scal);
//    algebra_base& left_smul_inplace(const coefficient& rhs);
//    algebra_base& right_smul_inplace(const coefficient& rhs);
    algebra_t& sdiv_inplace(const scalars::Scalar & rhs);
    algebra_t& mul_inplace(const algebra_t& rhs);

    // Hybrid in-place operations
    algebra_t& add_scal_mul(const algebra_t& arg, const scalars::Scalar & scal);
    algebra_t& add_scal_div(const algebra_t& arg, const scalars::Scalar & scal);
    algebra_t& sub_scal_mul(const algebra_t& arg, const scalars::Scalar & scal);
    algebra_t& sub_scal_div(const algebra_t& arg, const scalars::Scalar & scal);
    algebra_t& add_mul(const algebra_t& lhs, const algebra_t& rhs);
    algebra_t& sub_mul(const algebra_t& lhs, const algebra_t& rhs);
//    algebra_base& mul_left_smul(const algebra_base& arg, const coefficient& scal);
//    algebra_base& mul_right_smul(const algebra_base& arg, const coefficient& scal);
    algebra_t& mul_smul(const algebra_t& arg, const scalars::Scalar & scal);
    algebra_t& mul_sdiv(const algebra_t& arg, const scalars::Scalar & scal);


    std::ostream& print(std::ostream& os) const;

    std::ostream& operator<<(std::ostream& os) const
    {
        return this->print(os);
    }

    bool operator==(const algebra_t& other) const;
    bool operator!=(const algebra_t& other) const;
};


template <typename Interface>
class linear_operator_base
{
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
template <typename Algebra>
struct algebra_info
{
    using basis_type = typename Algebra::basis_type;
    using basis_traits = dtl::basis_info<basis_type>;
    using scalar_type = typename Algebra::scalar_type;
    using rational_type = scalar_type;
    using reference = scalar_type&;
    using const_reference = const scalar_type&;
    using pointer = scalar_type*;
    using const_pointer = const scalar_type*;

    static const scalars::ScalarType * ctype() noexcept
    { return ::esig::scalars::dtl::scalar_type_holder<scalar_type>::get_type(); }
    static constexpr VectorType vtype() noexcept
    { return VectorType::sparse; }
    static deg_t width(const Algebra* instance) noexcept
    { return instance->basis().width(); }
    static deg_t max_depth(const Algebra* instance) noexcept
    { return instance->basis().depth(); }

    static const basis_type& basis(const Algebra& instance) noexcept
    { return instance.basis(); }

    using this_key_type = typename Algebra::key_type;
    static this_key_type convert_key(const Algebra* instance, esig::key_type key) noexcept
    { return basis_traits::convert_key(instance->basis(), key); }

    static deg_t degree(const Algebra& instance) noexcept
    { return instance.degree(); }
    static deg_t degree(const Algebra* instance, esig::key_type key) noexcept
    { return instance->basis().degree(convert_key(instance, key)); }
    static deg_t native_degree(const Algebra* instance, this_key_type key)
    { return instance->basis().degree(key); }

    static key_type first_key(const Algebra*) noexcept
    { return 0; }
    static key_type last_key(const Algebra* instance) noexcept
    { return instance->basis().size(); }

    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;
    static const key_prod_container& key_product(const Algebra* inst, key_type k1, key_type k2) {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }
    static const key_prod_container& key_product(const Algebra* inst, const this_key_type& k1, const this_key_type& k2)
    {
        static const boost::container::small_vector<std::pair<key_type, int>, 0> null;
        return null;
    }

    static Algebra create_like(const Algebra& instance)
    {
        return Algebra();
    }
};






template <typename Interface>
std::ostream& operator<<(std::ostream& os, const algebra_base<Interface>& arg)
{
    return arg.print(os);
}


template<typename Algebra>
deg_t AlgebraInterface<Algebra>::degree() const {
    return 0;
}
template<typename Algebra>
deg_t AlgebraInterface<Algebra>::width() const {
    return 0;
}
template<typename Algebra>
deg_t AlgebraInterface<Algebra>::depth() const {
    return 0;
}
template<typename Algebra>
scalars::ScalarArray AlgebraInterface<Algebra>::dense_data() const {
    throw std::runtime_error("cannot retrieve dense data");
}

template<typename Algebra>
std::uintptr_t AlgebraInterface<Algebra>::id() const noexcept {
    return 0;
}
template<typename Algebra>
ImplementationType AlgebraInterface<Algebra>::type() const noexcept {
    return ImplementationType::owned;
}

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
void AlgebraInterface<Algebra>::add_scal_mul(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
    auto tmp = rhs.smul(coeff);
    this->add_inplace(*tmp);
}
template<typename Algebra>
void AlgebraInterface<Algebra>::sub_scal_mul(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
    auto tmp = rhs.smul(coeff);
    this->sub_inplace(*tmp);
}
template<typename Algebra>
void AlgebraInterface<Algebra>::add_scal_div(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
    auto tmp = rhs.sdiv(coeff);
    this->add_inplace(*tmp);
}
template<typename Algebra>
void AlgebraInterface<Algebra>::sub_scal_div(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
    auto tmp = rhs.sdiv(coeff);
    this->sub_inplace(*tmp);
}
template<typename Algebra>
void AlgebraInterface<Algebra>::add_mul(const AlgebraInterface &lhs, const AlgebraInterface &rhs) {
    auto tmp = lhs.mul(rhs);
    this->add_inplace(*tmp);
}
template<typename Algebra>
void AlgebraInterface<Algebra>::sub_mul(const AlgebraInterface &lhs, const AlgebraInterface &rhs) {
    auto tmp = lhs.mul(rhs);
    this->sub_inplace(*tmp);
}
template<typename Algebra>
void AlgebraInterface<Algebra>::mul_smul(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
    auto tmp = rhs.smul(coeff);
    this->mul_inplace(*tmp);
}
template<typename Algebra>
void AlgebraInterface<Algebra>::mul_sdiv(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
    auto tmp = rhs.sdiv(coeff);
    this->mul_inplace(*tmp);
}


namespace dtl {

bool fallback_equals(const algebra_iterator &lbegin, const algebra_iterator &lend,
                     const algebra_iterator &rbegin, const algebra_iterator &rend) noexcept;

template <typename Algebra, typename Interface, typename Fn>
Algebra fallback_multiply(const Algebra& lhs, const Interface& rhs, Fn op)
{
    auto result = algebra_info<Algebra>::create_like(lhs);
    fallback_multiplication(result, lhs, rhs, op);
    return result;
}

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
dimn_t algebra_implementation<Interface, Impl>::size() const {
    return m_data.size();
}
template<typename Interface, typename Impl>
deg_t algebra_implementation<Interface, Impl>::degree() const {
    return m_data.degree();
}
template<typename Interface, typename Impl>
deg_t algebra_implementation<Interface, Impl>::width() const {
    return algebra_info<Impl>::width(&m_data);
}
template<typename Interface, typename Impl>
deg_t algebra_implementation<Interface, Impl>::depth() const {
    return algebra_info<Impl>::max_depth(&m_data);
}
template<typename Interface, typename Impl>
VectorType algebra_implementation<Interface, Impl>::storage_type() const noexcept {
    return algebra_info<Impl>::vtype();
}
template<typename Interface, typename Impl>
const scalars::ScalarType *algebra_implementation<Interface, Impl>::coeff_type() const noexcept {
    return algebra_info<Impl>::ctype();
}
template<typename Interface, typename Impl>
scalars::Scalar algebra_implementation<Interface, Impl>::get(key_type key) const {
    using info_t = algebra_info<Impl>;
    auto akey = info_t::convert_key(&m_data, key);
    using ref_t = decltype(m_data[akey]);
    using trait = ::esig::scalars::dtl::scalar_type_trait<ref_t>;
    return trait::make(m_data[akey]);
}
template<typename Interface, typename Impl>
scalars::Scalar algebra_implementation<Interface, Impl>::get_mut(key_type key) {
    using info_t = algebra_info<Impl>;
    auto akey = info_t::convert_key(&m_data, key);
    using ref_t = decltype(m_data[akey]);
    using trait = ::esig::scalars::dtl::scalar_type_trait<ref_t>;
    return trait::make(m_data[akey]);
}
template<typename Interface, typename Impl>
algebra_iterator algebra_implementation<Interface, Impl>::begin() const {
    return algebra_iterator(&algebra_info<Impl>::basis(m_data), m_data.begin(), p_ctx);
}
template<typename Interface, typename Impl>
algebra_iterator algebra_implementation<Interface, Impl>::end() const {
    return algebra_iterator(&algebra_info<Impl>::basis(m_data), m_data.end(), p_ctx);
}
//template<typename Interface, typename Impl>
//dense_data_access_iterator algebra_implementation<Interface, Impl>::iterate_dense_components() const {
//    return dense_data_access_iterator(
////        dtl::dense_data_access_implementation<Impl>(m_data, algebra_info<Impl>::first_key(&m_data))
//        );
//}


template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::uminus() const {
    return algebra_t(-m_data, p_ctx);
}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::add(const algebra_interface_t &other) const {
    assert (id() != 0);
    if (id() == other.id()) {
        return algebra_t(m_data + cast<const Impl&>(other), p_ctx);
    }
    using ref_type = typename algebra_info<Impl>::reference;
    using cref_type = typename algebra_info<Impl>::const_reference;
    return algebra_t(fallback_binary_op(m_data, other,
            [](ref_type lhs, cref_type rhs) { lhs += rhs; }), p_ctx);
}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::sub(const algebra_interface_t &other) const {
    assert(id() != 0);
    if (id() == other.id()) {
        return algebra_t(m_data - cast<const Impl&>(other), p_ctx);
    }
    using ref_type = typename algebra_info<Impl>::reference;
    using cref_type = typename algebra_info<Impl>::const_reference;
    return algebra_t(fallback_binary_op(m_data, other,
                                        [](ref_type lhs, cref_type rhs) { lhs -= rhs; }), p_ctx);
}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::smul(const scalars::Scalar &scal) const {
    return algebra_t(m_data * scalars::scalar_cast<this_scalar_type>(scal), p_ctx);
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
typename Interface::algebra_t algebra_implementation<Interface, Impl>::sdiv(const scalars::Scalar &other) const {
    return algebra_t(m_data/scalars::scalar_cast<rational_type>(other), p_ctx);
}
template<typename Interface, typename Impl>
typename Interface::algebra_t algebra_implementation<Interface, Impl>::mul(const algebra_interface_t &other) const {
    assert(id() != 0);
    if (id() == other.id()) {
        return algebra_t(m_data * cast<const Impl&>(other), p_ctx);
    }
    return algebra_t(fallback_multiply(m_data, other,
                                       [](auto arg) { return arg; }), p_ctx);

}
template<typename Interface, typename Impl>
void algebra_implementation<Interface, Impl>::add_inplace(const algebra_interface_t &other) {
    assert(id() != 0);
    if (id() == other.id()) {
        m_data += cast<const Impl&>(other);
    } else {
        using ref_type = typename algebra_info<Impl>::reference;
        using cref_type = typename algebra_info<Impl>::const_reference;
        fallback_inplace_binary_op(m_data, other,
                                   [](ref_type lhs, cref_type rhs) { lhs += rhs; });
    }

}
template<typename Interface, typename Impl>
void algebra_implementation<Interface, Impl>::sub_inplace(const algebra_interface_t &other) {
    assert(id() != 0);
    if (id() == other.id()) {
        m_data -= cast<const Impl&>(other);
    } else {
        using ref_type = typename algebra_info<Impl>::reference;
        using cref_type = typename algebra_info<Impl>::const_reference;
        fallback_inplace_binary_op(m_data, other,
                                   [](ref_type lhs, cref_type rhs) { lhs -= rhs; });
    }

}
template<typename Interface, typename Impl>
void algebra_implementation<Interface, Impl>::smul_inplace(const scalars::Scalar &other) {
    m_data *= scalars::scalar_cast<this_scalar_type>(other);
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
void algebra_implementation<Interface, Impl>::sdiv_inplace(const scalars::Scalar &other) {
    m_data /= scalars::scalar_cast<rational_type>(other);
}
template<typename Interface, typename Impl>
void algebra_implementation<Interface, Impl>::mul_inplace(const algebra_interface_t &other) {
    assert(id() != 0);
    if (id() == other.id()) {
        m_data *= cast<const Impl&>(other);
    } else {
        fallback_inplace_multiplication(m_data, other, [](auto arg) { return arg; });
    }
}
template<typename Interface, typename Impl>
std::ostream &algebra_implementation<Interface, Impl>::print(std::ostream &os) const {
    return os << m_data;
}
template<typename Interface, typename Impl>
bool algebra_implementation<Interface, Impl>::equals(const algebra_interface_t &other) const {
    assert(id() != 0);

    if (id() == other.id()) {
        return m_data == cast<const Impl&>(other);
    }
    return fallback_equals(begin(), end(), other.begin(), other.end());
}

template<typename Interface, typename Impl>
std::uintptr_t algebra_implementation<Interface, Impl>::id() const noexcept {
    return (std::uintptr_t) p_ctx;
}

template<typename Interface, typename Impl>
borrowed_algebra_implementation<Interface, Impl>::borrowed_algebra_implementation(
    Impl *impl, borrowed_algebra_implementation::context_p ctx)
    : p_impl(impl), p_ctx(ctx)
{
    assert(p_impl != nullptr);
}
template<typename Interface, typename Impl>
dimn_t borrowed_algebra_implementation<Interface, Impl>::size() const {
    return p_impl->size();
}
template<typename Interface, typename Impl>
deg_t borrowed_algebra_implementation<Interface, Impl>::degree() const {
    return p_impl->degree();
}
template<typename Interface, typename Impl>
deg_t borrowed_algebra_implementation<Interface, Impl>::width() const {
    return algebra_info<Impl>::width(p_impl);
}
template<typename Interface, typename Impl>
deg_t borrowed_algebra_implementation<Interface, Impl>::depth() const {
    return algebra_info<Impl>::max_depth(p_impl);
}
template<typename Interface, typename Impl>
VectorType borrowed_algebra_implementation<Interface, Impl>::storage_type() const noexcept {
    return algebra_info<Impl>::vtype();
}
template<typename Interface, typename Impl>
const scalars::ScalarType *borrowed_algebra_implementation<Interface, Impl>::coeff_type() const noexcept {
    return algebra_info<Impl>::ctype();
}
template<typename Interface, typename Impl>
scalars::Scalar borrowed_algebra_implementation<Interface, Impl>::get(key_type key) const {
    using info_t = algebra_info<Impl>;
    using ref_t = decltype((*p_impl)[info_t::convert_key(p_impl, key)]);
    using trait = ::esig::scalars::dtl::scalar_type_trait<ref_t>;
    return trait::make((*p_impl)[info_t::convert_key(p_impl, key)]);
}
template<typename Interface, typename Impl>
scalars::Scalar borrowed_algebra_implementation<Interface, Impl>::get_mut(key_type key) {
    using info_t = algebra_info<Impl>;
    using ref_t = decltype((*p_impl)[info_t::convert_key(p_impl, key)]);
    using trait = ::esig::scalars::dtl::scalar_type_trait<ref_t>;
    return trait::make((*p_impl)[info_t::convert_key(p_impl, key)]);
}
template<typename Interface, typename Impl>
algebra_iterator borrowed_algebra_implementation<Interface, Impl>::begin() const {
    return algebra_iterator(p_impl->begin(), p_ctx);
}
template<typename Interface, typename Impl>
algebra_iterator borrowed_algebra_implementation<Interface, Impl>::end() const {
    return algebra_iterator(p_impl->end(), p_ctx);
}
//template<typename Interface, typename Impl>
//dense_data_access_iterator borrowed_algebra_implementation<Interface, Impl>::iterate_dense_components() const {
//    return dense_data_access_iterator(
//        dtl::dense_data_access_implementation<Impl>(*p_impl, algebra_info<Impl>::first_key(p_impl))
//        );
//}
template<typename Interface, typename Impl>
typename borrowed_algebra_implementation<Interface, Impl>::algebra_t
 borrowed_algebra_implementation<Interface, Impl>::uminus() const {
    return algebra_t(-(*p_impl), p_ctx);
}
template<typename Interface, typename Impl>
typename borrowed_algebra_implementation<Interface, Impl>::algebra_t
 borrowed_algebra_implementation<Interface, Impl>::add(const borrowed_algebra_implementation::algebra_interface_t &other) const {
    assert(id() != 0);
    assert(p_impl != nullptr);
    if (id() == other.id()) {
        return algebra_t((*p_impl) + alg_impl_t::template cast<const Impl &>(other), p_ctx);
    }
    using ref_type = typename Impl::reference;
    return algebra_t(fallback_binary_op(*p_impl, other, [](ref_type lhs, const auto& rhs) { lhs += rhs; }), p_ctx);
}
template<typename Interface, typename Impl>
typename borrowed_algebra_implementation<Interface, Impl>::algebra_t
 borrowed_algebra_implementation<Interface, Impl>::sub(const borrowed_algebra_implementation::algebra_interface_t &other) const {
    assert(id() != 0);
    assert(p_impl != nullptr);
    if (id() == other.id()) {
        return algebra_t((*p_impl) - alg_impl_t::template cast<const Impl&>(other), p_ctx);
    }
    using ref_type = typename Impl::reference;
    return algebra_t(fallback_binary_op(*p_impl, other, [](ref_type lhs, const auto& rhs) { lhs -= rhs; }), p_ctx);
}
template<typename Interface, typename Impl>
typename borrowed_algebra_implementation<Interface, Impl>::algebra_t
 borrowed_algebra_implementation<Interface, Impl>::smul(const scalars::Scalar &scal) const {
    return algebra_t((*p_impl) * scalars::scalar_cast<this_scalar_type>(scal), p_ctx);
}
template<typename Interface, typename Impl>
typename borrowed_algebra_implementation<Interface, Impl>::algebra_t
 borrowed_algebra_implementation<Interface, Impl>::sdiv(const scalars::Scalar &other) const {
    return algebra_t((*p_impl) / scalars::scalar_cast<rational_type>(other), p_ctx);
}
template<typename Interface, typename Impl>
typename borrowed_algebra_implementation<Interface, Impl>::algebra_t
 borrowed_algebra_implementation<Interface, Impl>::mul(const borrowed_algebra_implementation::algebra_interface_t &other) const {
    return algebra_t((*p_impl) * alg_impl_t::cast(other), p_impl);
}
template<typename Interface, typename Impl>
void borrowed_algebra_implementation<Interface, Impl>::add_inplace(
    const borrowed_algebra_implementation::algebra_interface_t &other) {
    (*p_impl) += alg_impl_t::cast(other);
}
template<typename Interface, typename Impl>
void borrowed_algebra_implementation<Interface, Impl>::sub_inplace(const borrowed_algebra_implementation::algebra_interface_t &other) {
    (*p_impl) -= alg_impl_t::cast(other);
}
template<typename Interface, typename Impl>
void borrowed_algebra_implementation<Interface, Impl>::smul_inplace(const scalars::Scalar &other) {
    (*p_impl) *= scalars::scalar_cast<this_scalar_type>(other);
}
template<typename Interface, typename Impl>
void borrowed_algebra_implementation<Interface, Impl>::sdiv_inplace(const scalars::Scalar &other) {
    (*p_impl) /= scalars::scalar_cast<rational_type>(other);
}
template<typename Interface, typename Impl>
void borrowed_algebra_implementation<Interface, Impl>::mul_inplace(const borrowed_algebra_implementation::algebra_interface_t &other) {
    (*p_impl) *= cast(other);
}
template<typename Interface, typename Impl>
std::ostream &borrowed_algebra_implementation<Interface, Impl>::print(std::ostream &os) const {
    return os << (*p_impl);
}
template<typename Interface, typename Impl>
bool borrowed_algebra_implementation<Interface, Impl>::equals(const borrowed_algebra_implementation::algebra_interface_t &other) const {
    return (*p_impl) == cast(other);
}

template<typename Interface, typename Impl>
std::uintptr_t borrowed_algebra_implementation<Interface, Impl>::id() const noexcept {
    return (std::uintptr_t) p_ctx;
}
template<typename Interface, typename Impl>
ImplementationType borrowed_algebra_implementation<Interface, Impl>::type() const noexcept {
    return ImplementationType::borrowed;
}

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
VectorType algebra_base<Interface>::storage_type() const noexcept {
    return p_impl->storage_type();
}
template<typename Interface>
const scalars::ScalarType * algebra_base<Interface>::coeff_type() const noexcept {
    return p_impl->coeff_type();
}
template<typename Interface>
scalars::Scalar algebra_base<Interface>::operator[](key_type k) const {
    return p_impl->get(k);
}
template<typename Interface>
scalars::Scalar algebra_base<Interface>::operator[](key_type k) {
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
//template<typename Interface>
//dense_data_access_iterator algebra_base<Interface>::iterate_dense_components() const noexcept {
//    return p_impl->iterate_dense_components();
//}
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
typename algebra_base<Interface>::algebra_t algebra_base<Interface>::smul(const scalars::Scalar &rhs) const {
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
typename algebra_base<Interface>::algebra_t algebra_base<Interface>::sdiv(const scalars::Scalar &scal) const {
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
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::smul_inplace(const scalars::Scalar &scal) {
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
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::sdiv_inplace(const scalars::Scalar & rhs) {
    p_impl->sdiv_inplace(rhs);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::mul_inplace(const algebra_t &rhs) {
    p_impl->mul_inplace(*rhs.p_impl);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::add_scal_mul(const algebra_t &arg, const scalars::Scalar &scal) {
    p_impl->add_scal_mul(*arg.p_impl, scal);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::add_scal_div(const algebra_t &arg, const scalars::Scalar &scal) {
    p_impl->add_scal_div(*arg.p_impl, scal);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::sub_scal_mul(const algebra_t &arg, const scalars::Scalar &scal) {
    p_impl->sub_scal_mul(*arg.p_impl, scal);
    return static_cast<algebra_t&>(*this);
}
template<typename Interface>
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::sub_scal_div(const algebra_t &arg, const scalars::Scalar &scal) {
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
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::mul_smul(const algebra_t &arg, const scalars::Scalar &scal) {
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
typename algebra_base<Interface>::algebra_t &algebra_base<Interface>::mul_sdiv(const algebra_t &arg, const scalars::Scalar &scal) {
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



namespace dtl {

template <typename Impl>
using key_value_buffer = std::vector<std::pair<
    typename algebra_info<Impl>::this_key_type,
    typename algebra_info<Impl>::scalar_type>>;

template <typename Impl>
using degree_range_buffer = std::vector<typename key_value_buffer<Impl>::const_iterator>;

template<typename Algebra, typename LhsIt, typename RhsIt, typename Op>
void fallback_multiplication_impl(Algebra &result,
                             const multiplication_data<LhsIt>& lhs,
                             const multiplication_data<RhsIt>& rhs,
                             Op op)
{
    using litraits = iterator_traits<LhsIt>;
    using ritraits = iterator_traits<RhsIt>;
    using info = algebra_info<Algebra>;

    auto max_deg = std::min(info::max_depth(&result), lhs.degree + rhs.degree);

    const auto* ref = &result;
    auto product = [ref](LhsIt lit, RhsIt rit) -> decltype(auto) {
      auto lkey = litraits::key(lit);
      auto rkey = ritraits::key(rit);
      return info::key_product(ref, lkey, rkey);
    };


    for (int out_deg = static_cast<int>(max_deg); out_deg > 0; --out_deg) {
        int lhs_max_deg = std::min(out_deg, static_cast<int>(lhs.degree));
        int lhs_min_deg = std::max(0, out_deg - static_cast<int>(rhs.degree));

        for (int lhs_d = lhs_max_deg; lhs_d >= lhs_min_deg; --lhs_d) {
            auto rhs_d = out_deg - lhs_d;

            for (auto lit = lhs.degree_ranges[lhs_d]; lit != lhs.degree_ranges[lhs_d + 1]; ++lit) {
                for (auto rit = rhs.degree_ranges[rhs_d]; rit != rhs.degree_ranges[rhs_d + 1]; ++rit) {
                    auto val = op(litraits::value(lit) * ritraits::value(rit));
                    for (auto prd : product(lit, rit)) {
                        result[info::convert_key(&result, prd.first)] += prd.second*val;
                    }
                }
            }
        }
    }
}


template <typename Impl>
void impl_to_buffer(key_value_buffer<Impl>& buffer,
                    degree_range_buffer<Impl>& degree_ranges,
                    const Impl& arg)
{
    using traits = iterator_traits<typename Impl::const_iterator>;
    buffer.reserve(arg.size());
    for (auto it = arg.begin(); it != arg.end(); ++it) {
       buffer.emplace_back(
            traits::key(it),
            traits::value(it)
            );
    }

    std::sort(buffer.begin(), buffer.end(), [](auto lpair, auto rpair) {
        return lpair.first < rpair.first;
    });

    degree_ranges = degree_range_buffer<Impl>(algebra_info<Impl>::max_depth(&arg)+2, buffer.end());
    deg_t deg = 0;
    for (auto it = buffer.begin(); it != buffer.end() && deg < degree_ranges.size(); ++it) {
        deg_t d = algebra_info<Impl>::native_degree(&arg, it->first);
        while (d > deg) {
            degree_ranges[deg++] = it;
        }
    }

}


template <typename Impl, typename Interface>
void interface_to_buffer(key_value_buffer<Impl>& buffer,
                         degree_range_buffer<Impl>& degree_ranges,
                         const Interface& arg, const Impl* rptr)
{
    using info = algebra_info<Impl>;
    using scalar_type = typename info::scalar_type;
    using pair_t = std::pair<typename info::this_key_type, scalar_type>;
    buffer.reserve(arg.size());

    for (auto it = arg.begin(); it != arg.end(); ++it) {
        buffer.emplace_back(info::convert_key(rptr, it->key()), scalars::scalar_cast<scalar_type>(it->value()));
    }

    std::sort(buffer.begin(), buffer.end(),
              [](auto lhs, auto rhs) { return lhs.first < rhs.first; });


    degree_ranges = degree_range_buffer<Impl>(arg.depth() + 2, buffer.end());
    deg_t deg = 0;
    for (auto it = buffer.begin(); it != buffer.end() && deg < degree_ranges.size(); ++it) {
        deg_t d = info::native_degree(rptr, it->first);
        while (d > deg) {
            degree_ranges[deg++] = it;
        }
    }
}



template<typename Interface, typename Impl, typename Fn>
void fallback_multiplication(Impl &result, const Interface &lhs, const Interface &rhs, Fn op)
{
    using info = algebra_info<Impl>;
    using iter = typename key_value_buffer<Impl>::const_iterator;
    key_value_buffer<Impl> lhs_buf, rhs_buf;

    multiplication_data<iter> lhs_data;
    interface_to_buffer(lhs_buf, lhs_data.degree_ranges, lhs, &result);
    lhs_data.begin = lhs_buf.begin();
    lhs_data.end = lhs_buf.end();
    lhs_data.degree = lhs.degree();

    multiplication_data<iter> rhs_data;
    interface_to_buffer<Impl>(rhs_buf, rhs_data.degree_ranges, rhs, &result);
    rhs_data.begin = rhs_buf.begin();
    rhs_data.end = rhs_buf.end();
    rhs_data.degree = rhs.degree();

    multiply_and_add(result, lhs_data, rhs_data, op);
}
template<typename Interface, typename Impl, typename Fn>
void fallback_multiplication(Impl &result, const Impl &lhs, const Interface &rhs, Fn op)
{
    using info = algebra_info<Impl>;
    using iter = typename key_value_buffer<Impl>::const_iterator;
    key_value_buffer<Impl> lhs_buf, rhs_buf;

    multiplication_data<iter> lhs_data;
    impl_to_buffer(lhs_buf, lhs_data.degree_ranges, lhs);
    lhs_data.begin = lhs_buf.begin();
    lhs_data.end = lhs_buf.end();
    lhs_data.degree = lhs.degree();

    multiplication_data<iter> rhs_data;
    interface_to_buffer<Impl>(rhs_buf, rhs_data.degree_ranges, rhs, &result);
    rhs_data.begin = rhs_buf.begin();
    rhs_data.end = rhs_buf.end();
    rhs_data.degree = rhs.degree();

    multiply_and_add(result, lhs_data, rhs_data, op);
}
template<typename Interface, typename Impl, typename Fn>
void fallback_multiplication(Impl &result, const Interface &lhs, const Impl &rhs, Fn op)
{
    using info = algebra_info<Impl>;
    using iter = typename key_value_buffer<Impl>::const_iterator;
    key_value_buffer<Impl> lhs_buf, rhs_buf;

    multiplication_data<iter> lhs_data;
    interface_to_buffer(lhs_buf, lhs_data.degree_ranges, lhs, &result);
    lhs_data.begin = lhs_buf.begin();
    lhs_data.end = lhs_buf.end();
    lhs_data.degree = lhs.degree();

    multiplication_data<iter> rhs_data;
    impl_to_buffer<Impl>(rhs_buf, rhs_data.degree_ranges, rhs);
    rhs_data.begin = rhs_buf.begin();
    rhs_data.end = rhs_buf.end();
    rhs_data.degree = rhs.degree();

    multiply_and_add(result, lhs_data, rhs_data, op);
}

template<typename Interface, typename Impl, typename Fn>
void fallback_inplace_multiplication(Impl &result, const Interface &rhs, Fn op)
{
    auto tmp = algebra_info<Impl>::create_like(result);
    fallback_multiplication(tmp, result, rhs, op);
    result = tmp;
}

template<typename Interface, typename Impl, typename Fn>
Impl fallback_binary_op(const Impl &lhs, const Interface &rhs, Fn op) {
    Impl result(lhs);
    fallback_inplace_binary_op(result, rhs, op);
    return result;
}
template<typename Interface, typename Impl, typename Fn>
void fallback_inplace_binary_op(Impl &lhs, const Interface &rhs, Fn op) {
    using scalar_type = typename algebra_info<Impl>::scalar_type;

    for (const auto& item : rhs) {
        op(lhs[algebra_info<Impl>::convert_key(&lhs, item.key())], scalars::scalar_cast<scalar_type>(item.value()));
    }
}

} // namespace dtl




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


#endif//ESIG_SRC_ALGEBRA_INCLUDE_ESIG_ALGEBRA_BASE_H_
