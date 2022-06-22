//
// Created by user on 21/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_SRC_DENSE_KV_ITERATOR_H_
#define ESIG_PATHS_SRC_ALGEBRA_SRC_DENSE_KV_ITERATOR_H_

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
    static coefficient value(const iter_type &iter)
    {
        using traits = coefficient_type_trait<decltype(iter.value())>;
        return traits::make(iter.value());
    }
    static bool equals(const iter_type &iter1, const iter_type &) { return iter1.finished(); }
};


} // namespace dtl
} // namespace algebra
} // namespace esig


#endif//ESIG_PATHS_SRC_ALGEBRA_SRC_DENSE_KV_ITERATOR_H_