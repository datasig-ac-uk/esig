//
// Created by user on 21/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_SRC_DENSE_KV_ITERATOR_H_
#define ESIG_PATHS_SRC_ALGEBRA_SRC_DENSE_KV_ITERATOR_H_

#include <esig/implementation_types.h>

namespace esig {
namespace algebra {
namespace dtl {


template <typename Vector>
class dense_kv_iterator
{
    using scalar_type = typename Vector::scalar_type;
    using basis_type = typename Vector::basis_type;

    const scalar_type* m_ptr;
    const scalar_type* m_end;
    key_type m_key;
    bool started = false;

public:

    dense_kv_iterator(const scalar_type* ptr, const scalar_type* end, key_type key=0)
        : m_ptr(ptr), m_end(end), m_key(key)
    {}

    bool advance() noexcept
    {
        if (!started) {
            started = true;
        } else {
            ++m_ptr;
            ++m_key;
        }
        return m_ptr != m_end;
    }

    const scalar_type& value() const noexcept
    {
        return *m_ptr;
    }

    const key_type& key() const noexcept
    {
        return m_key;
    }

    bool finished() const noexcept
    {
        return m_ptr == m_end;
    }



};


template <typename Vector>
struct iterator_helper<dense_kv_iterator<Vector>>
{
    using iter_type = dense_kv_iterator<Vector>;

    static void advance(iter_type &iter) { iter.advance(); }
    static key_type key(const iter_type &iter) { return iter.key(); }
    static scalars::scalar value(const iter_type &iter)
    {
        using traits = ::esig::scalars::dtl::scalar_type_trait<decltype(iter.value())>;
        return traits::make(iter.value());
    }
    static bool equals(const iter_type &iter1, const iter_type &) { return iter1.finished(); }
};



template <typename Iterator>
struct dense_iterator
{
    using traits = std::iterator_traits<Iterator>;
    using value_type = dense_iterator;
    using reference = dense_iterator&;
    using const_reference = const dense_iterator&;
    using iterator_category = std::forward_iterator_tag;
    using pointer = dense_iterator*;
    using const_pointer = const dense_iterator*;

    dense_iterator(Iterator it, key_type start = 0)
        : m_it(it), m_key(start)
    {}

    dense_iterator& operator++() noexcept
    {
        ++m_it;
        ++m_key;
        return *this;
    }
    const dense_iterator operator++(int) noexcept
    {
        dense_iterator result(*this);
        ++(*this);
        return result;
    }
    const_reference operator*() const noexcept
    { return *this; }
    const_pointer operator->() const noexcept
    { return this; }

    const key_type& key() const noexcept
    { return m_key; }
    const typename traits::value_type& value() const noexcept
    { return *m_it; }

    bool operator==(const dense_iterator& other) const noexcept
    { return m_it == other.m_it; }
    bool operator!=(const dense_iterator& other) const noexcept
    { return m_it != other.m_it; }

private:
    Iterator m_it;
    key_type m_key;
};

template <typename Iterator>
struct iterator_traits<dense_iterator<Iterator>>
{
    using iterator_t = dense_iterator<Iterator>;
    static const key_type& key(const iterator_t & it) noexcept
    { return it->key(); }
    static const typename Iterator::value_type&
    value(const iterator_t& it) noexcept
    { return it->value(); }
};

} // namespace dtl
} // namespace algebra
} // namespace esig


#endif//ESIG_PATHS_SRC_ALGEBRA_SRC_DENSE_KV_ITERATOR_H_
