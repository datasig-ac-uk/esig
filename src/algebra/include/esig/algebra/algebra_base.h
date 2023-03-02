#ifndef ESIG_ALGEBRA_ALGEBRA_BASE_H_
#define ESIG_ALGEBRA_ALGEBRA_BASE_H_

#include "algebra_fwd.h"

#include <memory>
#include <type_traits>

#include <esig/scalar.h>
#include <esig/scalar_array.h>

#include "context_fwd.h"
#include "iteration.h"

namespace esig {
namespace algebra {

namespace dtl {

struct AlgebraInterfaceTag {};
}// namespace dtl

template<typename Algebra>
struct AlgebraInterface : public dtl::AlgebraInterfaceTag {
    using algebra_t = Algebra;
    using const_iterator = algebra_iterator;
    using algebra_interface_t = AlgebraInterface;

    virtual ~AlgebraInterface() = default;

    virtual dimn_t size() const = 0;
    virtual deg_t degree() const { return 0; }
    virtual deg_t width() const { return 0; }
    virtual deg_t depth() const { return 0; };
    virtual VectorType storage_type() const noexcept = 0;
    virtual const esig::scalars::ScalarType *coeff_type() const noexcept = 0;

    // Element access
    virtual scalars::Scalar get(key_type key) const = 0;
    virtual scalars::Scalar get_mut(key_type key) = 0;

    // Iteration
    virtual algebra_iterator begin() const = 0;
    virtual algebra_iterator end() const = 0;

    virtual scalars::ScalarArray dense_data() const {
        throw std::runtime_error("cannot retrieve dense data");
    }
    //    virtual dense_data_access_iterator iterate_dense_components() const = 0;

    // Arithmetic
    virtual Algebra uminus() const = 0;
    virtual Algebra add(const AlgebraInterface &other) const = 0;
    virtual Algebra sub(const AlgebraInterface &other) const = 0;
    virtual Algebra smul(const scalars::Scalar &other) const = 0;
    //    virtual Algebra left_smul(const coefficient& other) const = 0;
    //    virtual Algebra right_smul(const coefficient& other) const = 0;
    virtual Algebra sdiv(const scalars::Scalar &other) const = 0;
    virtual Algebra mul(const AlgebraInterface &other) const = 0;

    // Inplace Arithmetic
    virtual void add_inplace(const AlgebraInterface &other) = 0;
    virtual void sub_inplace(const AlgebraInterface &other) = 0;
    virtual void smul_inplace(const scalars::Scalar &other) = 0;
    //    virtual void left_smul_inplace(const coefficient& other) = 0;
    //    virtual void right_smul_inplace(const coefficient& other) = 0;
    virtual void sdiv_inplace(const scalars::Scalar &other) = 0;
    virtual void mul_inplace(const AlgebraInterface &other) = 0;

    // Hybrid inplace arithmetic
    virtual void add_scal_mul(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
        auto tmp = rhs.smul(coeff);
        this->add_inplace(*tmp);
    }
    virtual void sub_scal_mul(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
        auto tmp = rhs.smul(coeff);
        this->sub_inplace(*tmp);
    }
    virtual void add_scal_div(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
        auto tmp = rhs.sdiv(coeff);
        this->add_inplace(*tmp);
    }
    virtual void sub_scal_div(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
        auto tmp = rhs.sdiv(coeff);
        this->sub_inplace(*tmp);
    }
    virtual void add_mul(const AlgebraInterface &lhs, const AlgebraInterface &rhs) {
        auto tmp = lhs.mul(rhs);
        this->add_inplace(*tmp);
    }
    virtual void sub_mul(const AlgebraInterface &lhs, const AlgebraInterface &rhs) {
        auto tmp = lhs.mul(rhs);
        this->sub_inplace(*tmp);
    }
    virtual void mul_smul(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
        auto tmp = rhs.smul(coeff);
        this->mul(*tmp);
    }
    virtual void mul_sdiv(const AlgebraInterface &rhs, const scalars::Scalar &coeff) {
        auto tmp = rhs.sdiv(coeff);
        this->mul(*tmp);
    }

    // Display methods
    virtual std::ostream &print(std::ostream &os) const = 0;

    // Equality testing
    virtual bool equals(const AlgebraInterface &other) const = 0;

    virtual std::uintptr_t id() const noexcept { return 0; }
    virtual ImplementationType type() const noexcept { return ImplementationType::owned; }
};


template<typename Impl>
class OwnedStorageModel;
template<typename Impl>
class BorrowedStorageModel;
template<typename, typename, template <typename> class>
class AlgebraImplementation;

namespace dtl {

template<typename Impl,
         template<typename, template<typename> class> class Wrapper = AlgebraImplementation>
using select_owned_borrowed_t = std::conditional_t<std::is_pointer<std::remove_reference_t<Impl>>::value,
                                                   Wrapper<std::remove_cv_t<std::remove_pointer_t<Impl>>, BorrowedStorageModel>,
                                                   Wrapper<std::remove_cv_t<std::remove_reference_t<Impl>>, OwnedStorageModel>>;


template <typename IFace>
struct with_interface {
    template <typename Impl, template <typename> class StorageModel>
    using type = AlgebraImplementation<IFace, Impl, StorageModel>;
};


}

template<typename Interface, template<typename, template<typename> class> class DerivedImpl = dtl::with_interface<Interface>::template type>
class AlgebraBase {
    explicit AlgebraBase(std::shared_ptr<Interface> impl)
        : p_impl(std::move(impl)) {}

protected:
    std::shared_ptr<Interface> p_impl;
    friend class algebra_base_access;

    friend class algebra_access<Interface>;
    friend class algebra_access<AlgebraInterface<typename Interface::algebra_t>>;

public:
    using interface_t = Interface;

    using algebra_t = typename Interface::algebra_t;
    using const_iterator = typename Interface::const_iterator;

    AlgebraBase() : p_impl(nullptr) {}

    template<typename Impl,
             typename = std::enable_if_t<
                 !std::is_same<
                     std::remove_cv_t<std::remove_reference_t<Impl>>, algebra_t>::value>>
    explicit AlgebraBase(Impl &&arg, const context *ctx)
        : p_impl(new dtl::select_owned_borrowed_t<Impl, DerivedImpl>(std::forward<Impl>(arg), ctx)) {
    }

    template<typename Impl, typename... Args>
    static std::enable_if_t<!std::is_base_of<Interface, Impl>::value, algebra_t>
    from_args(Args &&...args) {
        std::shared_ptr<interface_t> ptr(new dtl::select_owned_borrowed_t<Impl, DerivedImpl>(std::forward<Args>(args)...));
        return algebra_t(ptr);
    }

    template <typename Wrapper, typename... Args>
    static std::enable_if_t<std::is_base_of<Interface, Wrapper>::value, algebra_t>
    from_args(Args... args) {
        std::shared_ptr<interface_t> ptr(new Wrapper(std::forward<Args>(args)...));
        return algebra_t(ptr);
    }

    const Interface &operator*() const noexcept { return *p_impl; }
    Interface &operator*() noexcept { return *p_impl; };

    operator bool() const noexcept { return static_cast<bool>(p_impl); }// NOLINT(google-explicit-constructor)

    dimn_t size() const { return p_impl->size(); }
    deg_t width() const { return p_impl->width(); }
    deg_t depth() const { return p_impl->depth(); }
    deg_t degree() const { return p_impl->degree(); }
    VectorType storage_type() const noexcept { return p_impl->storage_type(); }
    const scalars::ScalarType *coeff_type() const noexcept { return p_impl->coeff_type(); }

    scalars::Scalar operator[](key_type k) const { return p_impl->get(k); }
    scalars::Scalar operator[](key_type k) { return p_impl->get_mut(k); }

    const_iterator begin() const { return p_impl->begin(); }
    const_iterator end() const { return p_impl->end(); }

    scalars::ScalarArray dense_data() const { return p_impl->dense_data(); }
    //    dense_data_access_iterator iterate_dense_components() const noexcept;

    // Binary operations
    algebra_t uminus() const { return p_impl->uminus(); }
    algebra_t add(const algebra_t &rhs) const { return p_impl->add(*rhs); }
    algebra_t sub(const algebra_t &rhs) const { return p_impl->sub(*rhs); };
    algebra_t smul(const scalars::Scalar &rhs) const { return p_impl->smul(rhs); };
    //    algebra_t left_smul(const coefficient& scal) const;
    //    algebra_t right_smul(const coefficient& scal) const;
    algebra_t sdiv(const scalars::Scalar &scal) const { return p_impl->sdiv(scal); }
    algebra_t mul(const algebra_t &rhs) const { return p_impl->mul(*rhs); };

    // In-place binary operations
    algebra_t &add_inplace(const algebra_t &rhs) {
        p_impl->add_inplace(*rhs);
        return static_cast<algebra_t &>(*this);
    }
    algebra_t &sub_inplace(const algebra_t &rhs) {
        p_impl->sub_inplace(*rhs);
        return static_cast<algebra_t &>(*this);
    }
    algebra_t &smul_inplace(const scalars::Scalar &scal) {
        p_impl->smul_inplace(scal);
        return static_cast<algebra_t &>(*this);
    }
    //    algebra_base& left_smul_inplace(const coefficient& rhs);
    //    algebra_base& right_smul_inplace(const coefficient& rhs);
    algebra_t &sdiv_inplace(const scalars::Scalar &rhs) {
        p_impl->sdiv_inplace(rhs);
        return static_cast<algebra_t &>(*this);
    }
    algebra_t &mul_inplace(const algebra_t &rhs) {
        p_impl->mul_inplace(*rhs);
        return static_cast<algebra_t &>(*this);
    }

    // Hybrid in-place operations
    algebra_t &add_scal_mul(const algebra_t &arg, const scalars::Scalar &scal) {
        p_impl->add_scal_mul(*arg, scal);
        return static_cast<algebra_t &>(*this);
    }
    algebra_t &add_scal_div(const algebra_t &arg, const scalars::Scalar &scal) {
        p_impl->add_scal_div(*arg, scal);
        return static_cast<algebra_t &>(*this);
    }
    algebra_t &sub_scal_mul(const algebra_t &arg, const scalars::Scalar &scal) {
        p_impl->sub_scal_mul(*arg, scal);
        return static_cast<algebra_t &>(*this);
    }
    algebra_t &sub_scal_div(const algebra_t &arg, const scalars::Scalar &scal) {
        p_impl->sub_scal_div(*arg, scal);
        return static_cast<algebra_t &>(*this);
    }
    algebra_t &add_mul(const algebra_t &lhs, const algebra_t &rhs) {
        p_impl->add_mul(*lhs, *rhs);
        return static_cast<algebra_t &>(*this);
    }
    algebra_t &sub_mul(const algebra_t &lhs, const algebra_t &rhs) {
        p_impl->sub_mul(*lhs, *rhs);
        return static_cast<algebra_t &>(*this);
    }
    //    algebra_base& mul_left_smul(const algebra_base& arg, const coefficient& scal);
    //    algebra_base& mul_right_smul(const algebra_base& arg, const coefficient& scal);
    algebra_t &mul_smul(const algebra_t &arg, const scalars::Scalar &scal) {
        p_impl->mul_smul(*arg, scal);
        return static_cast<algebra_t &>(*this);
    }
    algebra_t &mul_sdiv(const algebra_t &arg, const scalars::Scalar &scal) {
        p_impl->mul_sdiv(*arg, scal);
        return static_cast<algebra_t &>(*this);
    }

    std::ostream &print(std::ostream &os) const { return p_impl->print(os); }

    std::ostream &operator<<(std::ostream &os) const { return this->print(os); }

    bool operator==(const algebra_t &other) const { return p_impl->equals(*other); }
    bool operator!=(const algebra_t &other) const { return !p_impl->equals(*other); }
};
}// namespace algebra
}// namespace esig

#endif// ESIG_ALGEBRA_ALGEBRA_BASE_H_
