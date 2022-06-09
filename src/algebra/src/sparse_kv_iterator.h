//
// Created by sam on 23/03/2022.
//

#ifndef ESIG_PATHS_SPARSE_KV_ITERATOR_H
#define ESIG_PATHS_SPARSE_KV_ITERATOR_H

#include <esig/implementation_types.h>
#include <esig/algebra/iteration.h>

namespace esig {
namespace algebra {
namespace dtl {

template <typename Vector>
class sparse_kv_iterator
{
    using scalar_type = typename Vector::scalar_type;
    using iterator_t = typename Vector::const_iterator;

    iterator_t m_iterator;
    iterator_t m_end;
    bool started = false;

public:

    sparse_kv_iterator(iterator_t begin, iterator_t end)
        : m_iterator(begin), m_end(end)
    {
    }

    bool advance() noexcept
    {
        if (!started) {
            started = true;
        } else {
            ++m_iterator;
        }
        return m_iterator != m_end;
    }

    const scalar_type& value() const noexcept
    {
        return m_iterator->second;
    }
    const key_type& key() const noexcept
    {
        return m_iterator->first;
    }
};


template <typename Vector>
struct iterator_helper<sparse_kv_iterator<Vector>>
{
    using iter_type = sparse_kv_iterator<Vector>;

    static void advance(iter_type& iter) { iter.advance(); }
    static key_type key(const iter_type& iter) { return iter.key(); }
    static coefficient value(const iter_type& iter)
    {
        using traits = coefficient_type_trait<decltype(*iter)>;
        return traits::make(iter->value());
    }
    static bool equals(const iter_type& iter1, const iter_type&) { return iter1.finished(); }

};




} // namespace dtl
} // namespace algebra
} // namespace esig



#endif//ESIG_PATHS_SPARSE_KV_ITERATOR_H
