//
// Created by user on 07/03/2022.
//

#include "lie_basis.h"
#include "hall_basis_map.h"

#include <sstream>

namespace esig {
namespace algebra {

static dtl::hall_basis_map s_hall_basis_map;


std::string key_to_string_let_func(let_t let)
{
    return std::to_string(let);
}
std::string key_to_string_binop(const std::string& a, const std::string& b)
{
    std::stringstream ss;
    ss << '[' << a << ',' << b << ']';
    return ss.str();
}


lie_basis::lie_basis(deg_t width, deg_t depth)
    : m_width(width), m_depth(depth), m_hall_set(s_hall_basis_map.get(width, depth)),
      m_degree_impl(m_hall_set, &degree_let_func, &degree_binop_func),
      m_k2s_impl(m_hall_set, &key_to_string_let_func, &key_to_string_binop)
{
}
deg_t lie_basis::width() const noexcept
{
    return m_width;
}
deg_t lie_basis::depth() const noexcept
{
    return m_depth;
}
std::shared_ptr<hall_set> lie_basis::get_hall_set() const noexcept
{
    return m_hall_set;
}

std::string lie_basis::key_to_string(const key_type &key) const
{
    return "(" + m_k2s_impl(key) + ")";
}
key_type lie_basis::key_of_letter(let_t let) const noexcept
{
    return m_hall_set->key_of_letter(let);
}
bool lie_basis::letter(const key_type &type) const noexcept
{
    return m_hall_set->letter(type);
}
dimn_t lie_basis::size(int degree) const noexcept
{
    if (degree < 0) {
        return m_hall_set->size(m_depth);
    }
    return m_hall_set->size(degree);
}
dimn_t lie_basis::start_of_degree(deg_t deg) const noexcept
{
    if (deg == 0) return 0;
    return m_hall_set->size(deg-1);
}
key_type lie_basis::lparent(const key_type &key) const noexcept
{
    return (*m_hall_set)[key].first;
}
key_type lie_basis::rparent(const key_type &key) const noexcept
{
    return (*m_hall_set)[key].second;
}
let_t lie_basis::first_letter(const key_type &key) const noexcept
{
    return let_t((*m_hall_set)[key].first);
}
deg_t lie_basis::degree(const key_type &key) const
{
    return m_degree_impl(key);
}

typename lie_basis::product_type
lie_basis::prod_impl(const key_type &k1, const key_type &k2) const
{
    if (k1 > k2) {
        const auto& tmp = prod(k2, k1);
        product_type result;
        result.reserve(tmp.size());
        for (const auto& val : tmp) {
            result.emplace_back(val.first, -val.second);
        }
        return result;
    }

    typename hall_set::parent_type parents(k1, k2);
    auto in_hall_basis = (*m_hall_set)[parents];
    if (in_hall_basis != hall_set::root_element) {
        return { { in_hall_basis, 1 } };
    }

    auto k2_parents = (*m_hall_set)[k2];
    std::map<key_type, int> tmp;
    for (const auto& outer : prod(k1, k2_parents.first)) {
        for (const auto& inner : prod(outer.first, k2_parents.second)) {
            tmp[inner.first] += outer.second*inner.second;
        }
    }
    for (const auto& outer : prod(k1, k2_parents.second)) {
        for (const auto& inner : prod(outer.first, k2_parents.first)) {
            tmp[inner.first] -= outer.second*inner.second;
        }
    }

    return product_type(tmp.begin(), tmp.end());
}

const typename lie_basis::product_type&
lie_basis::prod(const key_type &k1, const key_type &k2) const
{
    static const product_type empty;
    if (k1 == k2) {
        return empty;
    }
    if (degree(k1) + degree(k2) > m_depth) {
        return empty;
    }


    std::lock_guard<std::recursive_mutex> access(prod_lock);
    typename hall_set::parent_type parents(k1, k2);
    auto& found = prod_cache[parents];
    if (!found.empty()) {
        return found;
    }

    return (found = prod_impl(k1, k2));
}


} // namespace algebra_old
} // namespace esig_paths
