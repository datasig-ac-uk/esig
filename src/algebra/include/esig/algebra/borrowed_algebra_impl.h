#ifndef ESIG_ALGEBRA_BORROWED_ALGEBRA_IMPL_H_
#define ESIG_ALGEBRA_BORROWED_ALGEBRA_IMPL_H_

#include "algebra_fwd.h"

#include <type_traits>

#include "algebra_base.h"


namespace esig { namespace algebra {

template <typename Interface, typename Impl>
class AlgebraImplementation;


template<typename Interface, typename Impl>
class borrowed_algebra_implementation : public Interface {
    using context_p = const context *;

    Impl *p_impl;
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
    using alg_impl_t = AlgebraImplementation<Interface, Impl>;

public:
    std::uintptr_t id() const noexcept override {
        return (std::uintptr_t) p_ctx;
    }
    ImplementationType type() const noexcept override {
        return ImplementationType::borrowed;
    }

    using scalar_type = typename algebra_info<impl_base_type>::scalar_type;
    using rational_type = typename algebra_info<impl_base_type>::rational_type;

    borrowed_algebra_implementation(Impl *impl, context_p ctx)
        : p_impl(impl), p_ctx(ctx)
    {}

    dimn_t size() const override { return p_impl->size(); }
    deg_t degree() const override { return p_impl->degree(); }
    deg_t width() const override { return algebra_info<Impl>::width(p_impl); }
    deg_t depth() const override { return algebra_info<Impl>::max_depth(p_impl); }
    VectorType storage_type() const noexcept override { return algebra_info<Impl>::vtype(); }
    const esig::scalars::ScalarType *coeff_type() const noexcept override { return algebra_info<Impl>::ctype(); }
    scalars::Scalar get(key_type key) const override {
        using info_t = algebra_info<Impl>;
        using ref_t = decltype((*p_impl)[info_t::convert_key(p_impl, key)]);
        using trait = ::esig::scalars::dtl::scalar_type_trait<ref_t>;
        return trait::make((*p_impl)[info_t::convert_key(p_impl, key)]);
    }
    scalars::Scalar get_mut(key_type key) override {
        using info_t = algebra_info<Impl>;
        using ref_t = decltype((*p_impl)[info_t::convert_key(p_impl, key)]);
        using trait = ::esig::scalars::dtl::scalar_type_trait<ref_t>;
        return trait::make((*p_impl)[info_t::convert_key(p_impl, key)]);
    }
    algebra_iterator begin() const override {
        return algebra_iterator(p_impl->begin(), p_ctx);
    }
    algebra_iterator end() const override {
        return algebra_iterator(p_impl->end(), p_ctx);
    }
    //    dense_data_access_iterator iterate_dense_components() const override;
    algebra_t uminus() const override {
        return algebra_t(-(*p_impl), p_ctx);
    }
    algebra_t add(const algebra_interface_t &other) const override {
        if (id() == other.id()) {
            return algebra_t((*p_impl) + alg_impl_t::template cast<const Impl &>(other), p_ctx);
        }
        using ref_type = typename Impl::reference;
        return algebra_t(fallback_binary_op(*p_impl, other, [](ref_type lhs, const auto &rhs) { lhs += rhs; }), p_ctx);
    }
    algebra_t sub(const algebra_interface_t &other) const override {
        if (id() == other.id()) {
            return algebra_t((*p_impl) - alg_impl_t::template cast<const Impl &>(other), p_ctx);
        }
        using ref_type = typename Impl::reference;
        return algebra_t(fallback_binary_op(*p_impl, other, [](ref_type lhs, const auto &rhs) { lhs -= rhs; }), p_ctx);
    }
    algebra_t smul(const scalars::Scalar &scal) const override {
        return algebra_t((*p_impl) * scalars::scalar_cast<scalar_type>(scal), p_ctx);
    }
    //    algebra_t left_smul(const coefficient &other) const override;
    //    algebra_t right_smul(const coefficient &other) const override;
    algebra_t sdiv(const scalars::Scalar &other) const override {
        return algebra_t((*p_impl) / scalars::scalar_cast<rational_type>(other), p_ctx);
    }
    algebra_t mul(const algebra_interface_t &other) const override {
        return algebra_t((*p_impl) * alg_impl_t::cast(other), p_impl);
    }
    void add_inplace(const algebra_interface_t &other) override {
        (*p_impl) += alg_impl_t::cast(other);
    }
    void sub_inplace(const algebra_interface_t &other) override {
        (*p_impl) -= alg_impl_t::cast(other);
    }
    void smul_inplace(const scalars::Scalar &other) override {
        (*p_impl) *= scalars::scalar_cast<scalar_type>(other);
    }
    //    void left_smul_inplace(const coefficient &other) override;
    //    void right_smul_inplace(const coefficient &other) override;
    void sdiv_inplace(const scalars::Scalar &other) override {
        (*p_impl) /= scalars::scalar_cast<rational_type>(other);
    }
    void mul_inplace(const algebra_interface_t &other) override {
        (*p_impl) *= cast(other);
    }
    std::ostream &print(std::ostream &os) const override {
        return os << (*p_impl);
    }
    bool equals(const algebra_interface_t &other) const override {
        return (*p_impl) == cast(other);
    }
};
}}

#endif // ESIG_ALGEBRA_BORROWED_ALGEBRA_IMPL_H_
