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
    dimn_t sz = 0;
    for (int i=0; i<=deg; ++i) {
        sz += m_powers->operator[](i);
    }
    return sz;
}
dimn_t tensor_basis::start_of_degree(deg_t deg) const noexcept
{
    if (deg == 0) return 0;
    return size(static_cast<int>(deg)-1);
}
key_type tensor_basis::lparent(const key_type &key) const noexcept
{
    if (key <= m_width) return key;
    auto pow = m_powers->operator[](degree(key) - 1);
    auto tmp = 1 + (key - 1) % pow;
    return (key - tmp) / pow;
}
key_type tensor_basis::rparent(const key_type &key) const noexcept
{
    if (key <= m_width) return 0;
    auto pow = m_powers->operator[](degree(key) - 1);
    return key % pow;
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


} // namespace algebra_old
} // namespace esig_paths
