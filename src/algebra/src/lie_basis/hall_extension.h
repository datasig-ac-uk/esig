//
// Created by sam on 07/03/2022.
//

#ifndef ESIG_PATHS_HALL_EXTENSION_H
#define ESIG_PATHS_HALL_EXTENSION_H

#include <esig/implementation_types.h>
#include "hall_set.h"

#include <functional>
#include <map>
#include <mutex>
#include <memory>


namespace esig {
namespace algebra {

template <typename Func, typename Binop>
class hall_extension
{
public:
    using key_type = typename hall_set::key_type;
    using out_type = decltype(std::declval < Func>()(std::declval < key_type>()));
private:
    std::shared_ptr<hall_set> m_hall_set;
    Func m_func;
    Binop m_binop;
    mutable std::map<key_type, out_type> m_cache;
    mutable std::recursive_mutex m_lock;
public:

    explicit hall_extension(std::shared_ptr<hall_set> hs, Func&& func, Binop&& binop);

    const out_type& operator()(const key_type& key) const;

};
template<typename Func, typename Binop>
hall_extension<Func, Binop>::hall_extension(std::shared_ptr<hall_set> hs, Func&& func, Binop&& binop)
    : m_hall_set(std::move(hs)), m_func(std::forward<Func>(func)), m_binop(std::forward<Binop>(binop))
{
}
template<typename Func, typename Binop>
const typename hall_extension<Func, Binop>::out_type&
hall_extension<Func, Binop>::operator()(const hall_extension::key_type &key) const
{
    std::lock_guard<std::recursive_mutex> access(m_lock);

    auto found = m_cache.find(key);
    if (found != m_cache.end()) {
        return found->second;
    }

    auto parents = (*m_hall_set)[key];
    return m_cache[key] = (m_hall_set->letter(key)) ? m_func(key)
                                                    : m_binop(operator()(parents.first), operator()(parents.second));
}


} // namespace algebra
} // namespace paths-old

#endif//ESIG_PATHS_HALL_EXTENSION_H
