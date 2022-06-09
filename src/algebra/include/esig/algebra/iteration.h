//
// Created by user on 21/03/2022.
//

#ifndef ESIG_ALGEBRA_ITERATION_H_
#define ESIG_ALGEBRA_ITERATION_H_

#include <esig/implementation_types.h>
#include <esig/algebra/esig_algebra_export.h>

#include <esig/algebra/coefficients.h>

#include <memory>

namespace esig {
namespace algebra {

class context;


struct ESIG_ALGEBRA_EXPORT algebra_iterator_item
{
    virtual ~algebra_iterator_item() = default;

    virtual key_type key() const noexcept = 0;
    virtual coefficient value() const noexcept = 0;
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


    template <typename Iter, typename=std::enable_if_t<
                !std::is_same<
                        std::remove_cv_t<std::remove_reference_t<Iter>>,
                        algebra_iterator>::value>>
    explicit algebra_iterator(Iter&& iter, const context* ctx);
    template <typename Iter, typename=std::enable_if_t<
                !std::is_same<
                        std::remove_cv_t<std::remove_reference_t<Iter>>,
                        algebra_iterator>::value>>
    explicit algebra_iterator(Iter&& iter);

    const context* get_context() const noexcept;
    algebra_iterator& operator++();
    reference operator*() const;
    pointer operator->() const;

    bool operator==(const algebra_iterator& other) const;
    bool operator!=(const algebra_iterator& other) const;
};

namespace dtl {

/*
 * The actual implementation of the iterator will usually depend on the
 * iterator type of the vector that produces it. This is handled by
 * creating an implementation of the algebra_iterator_interface with the
 * vector iterator type as a template parameter.
 */




template <typename Iter>
class algebra_iterator_implementation : public algebra_iterator_interface, public algebra_iterator_item
{
    Iter m_current;

public:

    algebra_iterator_implementation(const algebra_iterator_implementation& arg);
    algebra_iterator_implementation(algebra_iterator_implementation&& arg) noexcept;

    explicit algebra_iterator_implementation(Iter&& iter);
    void advance() override;
    const algebra_iterator_item &get() const override;
    pointer get_ptr() const override;
    bool equals(const algebra_iterator_interface &other) const noexcept override;
    key_type key() const noexcept override;
    coefficient value() const noexcept override;
};
} // namespace dtl




template<typename Iter, typename>
algebra_iterator::algebra_iterator(Iter &&iter, const context* ctx)
  : p_impl(new dtl::algebra_iterator_implementation<Iter>(std::forward<Iter>(iter))),
      p_ctx(ctx)
{
}

template<typename Iter, typename>
algebra_iterator::algebra_iterator(Iter &&iter)
  : p_impl(new dtl::algebra_iterator_implementation<Iter>(std::forward<Iter>(iter))),
      p_ctx(nullptr)
{
}


namespace dtl {

template <typename Iter>
struct iterator_helper
{
    static void advance(Iter& iter)
    {
        ++iter;
    }
    static key_type key(const Iter& iter)
    {
        return iter->first;
    }
    static coefficient value(const Iter& iter)
    {
        using trait = coefficient_type_trait<decltype(iter->second)>;
        return trait::make(iter->second);
    }
    static bool equals(const Iter& iter1, const Iter& iter2)
    {
        return iter1 == iter2;
    }
};


template<typename Iter>
algebra_iterator_implementation<Iter>::algebra_iterator_implementation(const algebra_iterator_implementation &arg)
    : m_current(arg.m_current)
{
}
template<typename Iter>
algebra_iterator_implementation<Iter>::algebra_iterator_implementation(algebra_iterator_implementation &&arg) noexcept
    : m_current(std::move(arg.m_current))
{
}

template<typename Iter>
algebra_iterator_implementation<Iter>::algebra_iterator_implementation(Iter &&iter)
    : m_current(std::forward<Iter>(iter))
{
}
template<typename Iter>
void algebra_iterator_implementation<Iter>::advance()
{
    iterator_helper<Iter>::advance(m_current);
}
template<typename Iter>
const algebra_iterator_item &algebra_iterator_implementation<Iter>::get() const
{
    return *this;
}
template<typename Iter>
algebra_iterator_interface::pointer algebra_iterator_implementation<Iter>::get_ptr() const
{
  return this;
}
template<typename Iter>
bool algebra_iterator_implementation<Iter>::equals(const algebra_iterator_interface &other) const noexcept
{
  return iterator_helper<Iter>::equals(m_current, dynamic_cast<const algebra_iterator_implementation&>(other).m_current);
}
template<typename Iter>
key_type algebra_iterator_implementation<Iter>::key() const noexcept
{
    return iterator_helper<Iter>::key(m_current);
}
template<typename Iter>
coefficient algebra_iterator_implementation<Iter>::value() const noexcept
{
    return iterator_helper<Iter>::value(m_current);
}
} // namespace dtl


} // namespace algebra
} // namespace esig



#endif//ESIG_ALGEBRA_ITERATION_H_
