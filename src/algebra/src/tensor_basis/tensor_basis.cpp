//
// Created by sam on 07/03/2022.
//

#include "tensor_basis.h"
#include <mutex>
#include <vector>
#include <map>
#include <memory>
#include <cassert>
#include <sstream>

namespace esig {
namespace algebra {

namespace {

class tensor_powers_map
{
    std::map<deg_t, std::shared_ptr<std::vector<dimn_t>>> m_data;
    std::mutex access;

public:

    std::shared_ptr<std::vector<dimn_t>> get(deg_t width, deg_t depth);

};

std::shared_ptr<std::vector<dimn_t>> tensor_powers_map::get(deg_t width, deg_t depth)
{
    std::lock_guard<std::mutex> lock(access);
    auto& found = m_data[width];
    if (!found) {
        found.reset(new std::vector<dimn_t> {dimn_t(1)});
    }
    if (found->size() <= depth) {
        dimn_t dwidth (width);
        found->reserve(depth+1);
        for (auto i=found->size(); i<=depth; ++i) {
            found->push_back(found->back()*dwidth);
        }
    }
    return found;
}


constexpr deg_t logn(dimn_t num, dimn_t base) noexcept
{
    return (num < base) ? 0 : (logn(num / base, base) + 1);
}


}

static tensor_powers_map s_tensor_size_map;


tensor_basis::tensor_basis(deg_t width, deg_t depth)
    : m_width(width), m_depth(depth), m_powers(s_tensor_size_map.get(width, depth))
{
}
deg_t tensor_basis::width() const noexcept
{
    return m_width;
}
deg_t tensor_basis::depth() const noexcept
{
    return m_depth;
}
std::string tensor_basis::key_to_string(const key_type &key) const
{
    std::stringstream ss;
    ss << '(';
    auto sz = degree(key);
    auto tmp = key - tensor_basis::start_of_degree(sz);
    while (sz) {
        auto pow = (*m_powers)[sz-1];
        auto first_letter = 1 + tmp / pow;
        tmp = tmp % pow;
        ss << first_letter;

        sz -= 1;
        if (sz > 0) ss << ',';

    }
    ss << ')';
    return ss.str();
}
key_type tensor_basis::key_of_letter(let_t let) const noexcept
{
    assert(letter(let));
    return key_type(let);
}
bool tensor_basis::letter(const key_type &key) const noexcept
{
    return key > 0 && key <= m_width;
}
dimn_t tensor_basis::size(int deg) const noexcept
{
    if (deg < 0) {
        deg = static_cast<int>(m_depth);
    } else if (deg == 0) {
        return 1;
    }
    dimn_t sz = 0;
    for (int i=0; i<=deg; ++i) {
        sz += m_powers->operator[](i);
    }
    return sz;
}
dimn_t tensor_basis::start_of_degree(deg_t deg) const noexcept
{
    if (deg == 0) return 0;
    if (deg == 1) return 1;
    if (deg == 2) return 1 + m_width;
    return size(static_cast<int>(deg)-1);
}
key_type tensor_basis::lparent(const key_type &key) const noexcept
{
    if (key <= m_width) return key;
    auto deg = degree(key);
    auto sod = start_of_degree(deg);
    auto adjusted = key - sod;
    auto pow = m_powers->operator[](deg-1);
    return 1 + (adjusted / pow);
}
key_type tensor_basis::rparent(const key_type &key) const noexcept
{
    if (key <= m_width) return 0;
    auto deg = degree(key);
    auto adjusted = key - start_of_degree(deg);
    auto pow = m_powers->operator[](deg - 1);
    return (adjusted % pow) + start_of_degree(deg-1);
}

let_t tensor_basis::first_letter(const key_type &key) const noexcept
{
    return static_cast<let_t>(lparent(key));
}
deg_t tensor_basis::degree(const key_type &key) const
{
    if (key == 0) {
        return 0;
    } else if (key <= m_width) {
        return 1;
    } else {
        auto thrsh = 1 + m_width;
        deg_t deg = 1;
        while (key >= thrsh) {
            thrsh += (*m_powers)[++deg];
        }
        return deg;
//        return logn(key, static_cast<dimn_t>(m_width));
    }
}

const std::vector<dimn_t> &tensor_basis::powers() const noexcept
{
    return *m_powers;
}

const tensor_basis::product_type &tensor_basis::prod(key_type k1, key_type k2) const {
    auto deg1 = degree(k1);
    auto deg2 = degree(k2);

    static const product_type empty;
    if (deg1 + deg2 > m_depth) {
        return empty;
    }

    product_type& result = m_cache[{k1, k2}];
    if (result.empty()) {
        assert(k1 >= start_of_degree(deg1));
        assert(k2 >= start_of_degree(deg2));

        auto left_adj = k1 - start_of_degree(deg1);
        auto right_adj = k2 - start_of_degree(deg2);

        auto adj_prod = left_adj * (*m_powers)[deg2] + right_adj;
        result.emplace_back(adj_prod + start_of_degree(deg1 + deg2), 1);
    }

    return result;
}

} // namespace algebra_old
} // namespace esig_paths
