#ifndef ESIG_ALGEBRA_ALGEBRA_IMPL_H_
#define ESIG_ALGEBRA_ALGEBRA_IMPL_H_

#include "algebra_base.h"
#include <boost/type_traits.hpp>
#include <boost/type_traits/is_detected.hpp>
#include <type_traits>

namespace esig {
namespace algebra {

template<typename Interface, typename Impl>
class ImplAccessLayer : public Interface {
public:
    virtual Impl &get_data() noexcept = 0;
    virtual const Impl &get_data() const noexcept = 0;
};


template <typename Impl, typename Interface>
boost::copy_cv_t<Impl, Interface>&
cast_algebra(Interface& arg) noexcept {
    static_assert(std::is_base_of<dtl::AlgebraInterfaceTag, Interface>::value,
        "casting to algebra implementation is only possible for interfaces");
    return static_cast<boost::copy_cv_t<ImplAccessLayer<Interface, Impl>, Interface>& >(arg).get_data();
}


template<typename Impl>
class OwnedStorageModel  {
    Impl m_data;
protected:
    static constexpr ImplementationType s_type = ImplementationType::owned;

    explicit OwnedStorageModel(Impl &&arg) : m_data(std::move(arg)) {}
    explicit OwnedStorageModel(const Impl &arg) : m_data(arg) {}
    explicit OwnedStorageModel(Impl *arg) : m_data(*arg) {}

    Impl &data() noexcept { return m_data; }
    const Impl &data() const noexcept { return m_data; }

};

template<typename Impl>
class BorrowedStorageModel  {
    Impl *p_data;


protected:
    static constexpr ImplementationType s_type = ImplementationType::borrowed;

    explicit BorrowedStorageModel(Impl *arg) : p_data(arg) {}

    Impl &data() noexcept { return *p_data; }
    const Impl &data() const noexcept { return *p_data; }

};





template<typename Interface, typename Impl, template<typename> class StorageModel>
class AlgebraImplementation
    : protected StorageModel<Impl>,
      public ImplAccessLayer<Interface, Impl> {
    using storage_base_t = StorageModel<Impl>;

protected:
    using context_p = const context *;
    context_p p_ctx;

    static_assert(std::is_base_of<dtl::AlgebraInterfaceTag, Interface>::value,
                  "algebra_interface must be an accessible base of Interface");

public:
    using interface_t = Interface;
    using algebra_t = typename Interface::algebra_t;
    using algebra_interface_t = typename Interface::algebra_interface_t;

    friend class ::esig::algebra::algebra_access<Interface>;
    using borrowed_alg_impl_t = AlgebraImplementation<Interface, Impl, BorrowedStorageModel>;

public:
    using scalar_type = typename algebra_info<Impl>::scalar_type;
    using rational_type = typename algebra_info<Impl>::rational_type;

protected:
    using storage_base_t::data;

public:
    Impl &get_data() noexcept override {
        return data();
    }
    const Impl &get_data() const noexcept override {
        return data();
    }
protected:
    template<typename T>
    static std::add_lvalue_reference_t<T> cast(boost::copy_cv_t<algebra_interface_t, std::remove_reference_t<T>> &arg) noexcept {
        return static_cast<boost::copy_cv_t<ImplAccessLayer<Interface, Impl>,
            std::remove_reference_t<T>>&>(arg).get_data();
    }

    template <typename T>
    static std::add_pointer_t<T> cast(boost::copy_cv_t<algebra_interface_t, std::remove_pointer_t<T>>* arg) noexcept {
        return &static_cast<boost::copy_cv_t<ImplAccessLayer<Interface, Impl>,
            std::remove_pointer_t<T>>&>(*arg).get_data();
    }

public:
    std::uintptr_t id() const noexcept override { return (std::uintptr_t)p_ctx; }

    ImplementationType type() const noexcept override {
        return AlgebraImplementation::s_type;
    }


    template<typename Arg>
    explicit AlgebraImplementation(Arg &&arg, context_p ctx)
         : storage_base_t(std::forward<Arg>(arg)), p_ctx(ctx) {}

    dimn_t size() const override { return data().size(); }
    deg_t degree() const override { return data().degree(); };
    deg_t width() const override { return algebra_info<Impl>::width(&data()); }
    deg_t depth() const override { return algebra_info<Impl>::max_depth(&data()); }
    VectorType storage_type() const noexcept override { return algebra_info<Impl>::vtype(); }
    const esig::scalars::ScalarType *coeff_type() const noexcept override { return algebra_info<Impl>::ctype(); }
    scalars::Scalar get(key_type key) const override {
        using info_t = algebra_info<Impl>;
        auto akey = info_t::convert_key(&data(), key);
        using ref_t = decltype(data()[akey]);
        using trait = ::esig::scalars::dtl::scalar_type_trait<ref_t>;
        return trait::make(data()[akey]);
    }
    scalars::Scalar get_mut(key_type key) override {
        using info_t = algebra_info<Impl>;
        auto akey = info_t::convert_key(&data(), key);
        using ref_t = decltype(data()[akey]);
        using trait = ::esig::scalars::dtl::scalar_type_trait<ref_t>;
        return trait::make(data()[akey]);
    }
    algebra_iterator begin() const override {
        return algebra_iterator(&algebra_info<Impl>::basis(data()), data().begin(), p_ctx);
    }
    algebra_iterator end() const override {
        return algebra_iterator(&algebra_info<Impl>::basis(data()), data().end(), p_ctx);
    }

private:
    template<typename T>
    using has_as_ptr_t = decltype(std::declval<const T &>().as_ptr());

    template<typename B, typename = std::enable_if_t<boost::is_detected<has_as_ptr_t, B>::value>>
    scalars::ScalarArray dense_data_impl(const B &data) const {
        return {data.as_ptr(), coeff_type(), data.dimension()};
    }

    scalars::ScalarArray dense_data_impl(...) const {
        return Interface::dense_data();
    }

public:
    scalars::ScalarArray dense_data() const override {
        return dense_data_impl(data().base_vector());
    }
    //    dense_data_access_iterator iterate_dense_components() const override;
    algebra_t uminus() const override { return algebra_t(-data(), p_ctx); }
    algebra_t add(const algebra_interface_t &other) const override {
        return algebra_t(data() + cast<const Impl &>(other), p_ctx);
    }
    algebra_t sub(const algebra_interface_t &other) const override {
        return algebra_t(data() - cast<const Impl &>(other), p_ctx);
    }
    algebra_t smul(const scalars::Scalar &scal) const override {
        return algebra_t(data() * scalars::scalar_cast<scalar_type>(scal), p_ctx);
    }
    //    algebra_t left_smul(const coefficient &other) const override;
    //    algebra_t right_smul(const coefficient &other) const override;
    algebra_t sdiv(const scalars::Scalar &other) const override {
        return algebra_t(data() / scalars::scalar_cast<rational_type>(other), p_ctx);
    }
    algebra_t mul(const algebra_interface_t &other) const override {
        return algebra_t(data() * cast<const Impl &>(other), p_ctx);
    }
    void add_inplace(const algebra_interface_t &other) override {
        data() += cast<const Impl &>(other);
    }

    void sub_inplace(const algebra_interface_t &other) override {
        data() -= cast<const Impl &>(other);
    }
    void smul_inplace(const scalars::Scalar &other) override {
        data() *= scalars::scalar_cast<scalar_type>(other);
    }
    //    void left_smul_inplace(const coefficient &other) override;
    //    void right_smul_inplace(const coefficient &other) override;
    void sdiv_inplace(const scalars::Scalar &other) override {
        data() /= scalars::scalar_cast<rational_type>(other);
    }
    void mul_inplace(const algebra_interface_t &other) override {
        data() *= cast<const Impl &>(other);
    }
    std::ostream &print(std::ostream &os) const override {
        return os << data();
    }
    bool equals(const algebra_interface_t &other) const override {
        return data() == cast<const Impl &>(other);
    }
};

}// namespace algebra
}// namespace esig
#endif// ESIG_ALGEBRA_ALGEBRA_IMPL_H_
