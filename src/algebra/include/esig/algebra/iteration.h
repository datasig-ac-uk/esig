//
// Created by user on 21/03/2022.
//

#ifndef ESIG_ALGEBRA_ITERATION_H_
#define ESIG_ALGEBRA_ITERATION_H_

#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>
#include <esig/algebra/algebra_traits.h>
#include <esig/scalars.h>

#include <memory>

namespace esig {
namespace algebra {

class context;


struct ESIG_ALGEBRA_EXPORT algebra_iterator_item
{
    virtual ~algebra_iterator_item() = default;

    virtual key_type key() const noexcept = 0;
    virtual scalars::Scalar value() const noexcept = 0;
};


struct ESIG_ALGEBRA_EXPORT algebra_iterator_interface
{
    virtual ~algebra_iterator_interface() = default;

    using value_type = algebra_iterator_item;
    using reference = const algebra_iterator_item&;
    using pointer = const algebra_iterator_item*;

    virtual void advance() = 0;
    virtual reference get() const = 0;
    virtual pointer get_ptr() const = 0;

    virtual bool equals(const algebra_iterator_interface& other) const noexcept = 0;

};

class ESIG_ALGEBRA_EXPORT algebra_iterator
{
   std::shared_ptr<algebra_iterator_interface> p_impl;
   const context* p_ctx;
public:
    using value_type = algebra_iterator_item;
    using reference = const algebra_iterator_item&;
    using pointer = const algebra_iterator_item*;

    algebra_iterator();
    algebra_iterator(const algebra_iterator&);
    algebra_iterator(algebra_iterator&&) noexcept;


    template <typename Basis, typename Iter, typename=std::enable_if_t<
                !std::is_same<
                        std::remove_cv_t<std::remove_reference_t<Iter>>,
                        algebra_iterator>::value>>
    explicit algebra_iterator(const Basis* basis, Iter&& iter, const context* ctx);
    template <typename Basis, typename Iter, typename=std::enable_if_t<
                !std::is_same<
                        std::remove_cv_t<std::remove_reference_t<Iter>>,
                        algebra_iterator>::value>>
    explicit algebra_iterator(const Basis* basis, Iter&& iter);

    const context* get_context() const noexcept;
    algebra_iterator& operator++();
    reference operator*() const;
    pointer operator->() const;

    bool operator==(const algebra_iterator& other) const;
    bool operator!=(const algebra_iterator& other) const;
};



class ESIG_ALGEBRA_EXPORT dense_data_access_item
{
    key_type m_start_key;
    const void* m_begin;
    const void* m_end;
public:
    dense_data_access_item(key_type start_key, const void* begin, const void* end);

    inline operator bool() const noexcept
    { return m_begin != nullptr; }

    inline const void* begin() const noexcept
    { return m_begin; }
    inline const void* end() const noexcept
    { return m_end; }
    inline key_type first_key() const noexcept
    { return m_start_key; }
};

struct ESIG_ALGEBRA_EXPORT dense_data_access_interface
{
    virtual ~dense_data_access_interface() = default;

    /// Advance and return the next access item; begin should be nullptr to
    /// stop iteration
    virtual dense_data_access_item next() = 0;
};

class ESIG_ALGEBRA_EXPORT dense_data_access_iterator
{
    std::unique_ptr<dense_data_access_interface> p_impl;
public:

    dense_data_access_iterator(dense_data_access_iterator&& other) noexcept = default;

    template <typename Impl>
    explicit dense_data_access_iterator(Impl&& impl);

    inline dense_data_access_item next()
    { return p_impl->next(); }

};


namespace dtl {

/*
 * The actual implementation of the iterator will usually depend on the
 * iterator type of the vector that produces it. This is handled by
 * creating an implementation of the algebra_iterator_interface with the
 * vector iterator type as a template parameter.
 */


template<typename Iterator>
struct iterator_traits {
    static auto key(Iterator it) noexcept -> decltype(it->first)
    { return it->first; }
    static auto
    value(Iterator it) noexcept -> decltype(it->second)
    { return it->second; }
};


template <typename Basis, typename Iter>
class algebra_iterator_implementation : public algebra_iterator_interface, public algebra_iterator_item
{
    Iter m_current;
    const Basis* p_basis;

    using basis_traits = basis_info<Basis>;
    using iter_traits = iterator_traits<Iter>;

public:

    algebra_iterator_implementation(const algebra_iterator_implementation& arg)
        : m_current(arg.m_current), p_basis(arg.p_basis)
    {
    }
    algebra_iterator_implementation(algebra_iterator_implementation&& arg) noexcept
        : m_current(std::move(arg.m_current)), p_basis(arg.p_basis)
    {
    }

    explicit algebra_iterator_implementation(const Basis* basis, Iter&& iter)
        : m_current(std::move(iter)), p_basis(basis)
    {}

    void advance() override { ++m_current; }
    const algebra_iterator_item &get() const override { return *this; }
    pointer get_ptr() const override { return this; };
    bool equals(const algebra_iterator_interface &other) const noexcept override
    { return m_current == static_cast<const algebra_iterator_implementation&>(other).m_current; }
    key_type key() const noexcept override
    {
        return basis_traits::convert_key(*p_basis, iter_traits::key(m_current));
    }
    scalars::Scalar value() const noexcept override
    {
        using trait = ::esig::scalars::dtl::scalar_type_trait<decltype(iter_traits::value(m_current))>;
        return trait::make(iter_traits::value(m_current));
    }
};
} // namespace dtl




template<typename Basis, typename Iter, typename>
algebra_iterator::algebra_iterator(const Basis* basis, Iter &&iter, const context* ctx)
  : p_impl(new dtl::algebra_iterator_implementation<Basis, Iter>(basis, std::forward<Iter>(iter))),
      p_ctx(ctx)
{
}

template<typename Basis, typename Iter, typename>
algebra_iterator::algebra_iterator(const Basis* basis, Iter &&iter)
  : p_impl(new dtl::algebra_iterator_implementation<Basis, Iter>(basis, std::forward<Iter>(iter))),
      p_ctx(nullptr)
{
}


namespace dtl {


template <typename Algebra>
class dense_data_access_implementation;

//template <typename Algebra>
//class dense_data_access_implementation : public dense_data_access_interface
//{
//    key_type m_current_key;
//    const Algebra& m_alg;
//
//    using scalar_type = typename algebra_info<Algebra>::scalar_type;
//
//    key_type advance_key(const void* begin, const void* end)
//    {
//        const auto* b = reinterpret_cast<const scalar_type*>(begin);
//        const auto* e = reinterpret_cast<const scalar_type*>(end);
//        auto key = m_current_key;
//        m_current_key += static_cast<key_type>(e - b);
//        return key;
//    }
//
//public:
//
//    dense_data_access_implementation(const Algebra& alg, key_type start) : m_alg(alg), m_current_key(start)
//    {}
//
//    dense_data_access_item next() override {
//        auto ptrs = dense_data_access<Algebra>::starting_at(m_alg, m_current_key);
//        auto key = advance_key(ptrs.first, ptrs.second);
//        return dense_data_access_item(key, ptrs.first, ptrs.second);
//    }
//};




} // namespace dtl

template<typename Impl>
dense_data_access_iterator::dense_data_access_iterator(Impl&& impl)
    : p_impl(std::unique_ptr<dense_data_access_interface>(new Impl(std::forward<Impl>(impl))))
{
}

} // namespace algebra
} // namespace esig



#endif//ESIG_ALGEBRA_ITERATION_H_
