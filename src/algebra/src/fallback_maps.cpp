//
// Created by user on 05/07/22.
//
#include "fallback_maps.h"


using esig::key_type;

esig::algebra::dtl::expand_binop::expand_binop(const esig::algebra::tensor_basis *arg)
    : m_basis(arg)
{
}
esig::algebra::dtl::expand_binop::result_t esig::algebra::dtl::expand_binop::operator()(const esig::algebra::dtl::expand_binop::result_t &a, const esig::algebra::dtl::expand_binop::result_t &b) const {
    const auto &powers = m_basis->powers();
    std::map<key_type, int> tmp;
    for (const auto &lhs : a) {
        for (const auto &rhs : b) {
            auto lhs_deg = m_basis->degree(lhs.first);
            auto rhs_deg = m_basis->degree(rhs.first);
            auto lhs_shift = powers[lhs_deg];
            auto rhs_shift = powers[rhs_deg];

            tmp[lhs.first * rhs_shift + rhs.first] += lhs.second * rhs.second;
            tmp[rhs.first * lhs_shift + lhs.first] -= rhs.second * lhs.second;
        }
    }
    return {tmp.begin(), tmp.end()};
}

esig::algebra::maps::maps(std::shared_ptr<const tensor_basis> tbasis, std::shared_ptr<const lie_basis> lbasis)
    : p_tensor_basis(std::move(tbasis)), p_lie_basis(std::move(lbasis)),
      expand(p_lie_basis->get_hall_set(), &maps::expand_func, dtl::expand_binop(p_tensor_basis.get()))
{
}
std::vector<std::pair<key_type, int>>
esig::algebra::maps::expand_func(let_t letter) {
    return {{letter, 1}};
}

namespace esig {
namespace algebra {

class lie_product_operator {
    const lie_basis& m_basis;
public:

    using pair_type = std::pair<key_type, int>;
    using lie_t = std::vector<pair_type>;

    lie_product_operator(const lie_basis& basis) : m_basis(basis)
    {}

    lie_t operator()(const lie_t& a, const lie_t& b) const
    {
        auto max_deg = m_basis.depth();

        std::unordered_map<key_type, int> tmp;
        for (auto& x : a) {
            auto x_degree = m_basis.degree(x.first);
            for (auto& y : b) {
                if (x_degree + m_basis.degree(y.first) <= max_deg) {
                    for (auto z : m_basis.prod(x.first, y.first)) {
                        tmp[z.first] += z.second*x.second*y.second;
                    }
                }
            }
        }
        lie_t result(tmp.begin(), tmp.end());
        std::sort(result.begin(), result.end(), [](const pair_type & p1, const pair_type & p2) { return p1.first < p2.first; });

        return result;
    }


};
} // namespace algebra
} // namespace esig



const std::vector<std::pair<key_type, int>> &
esig::algebra::maps::rbracket(key_type key) const {
    auto width = p_tensor_basis->width();

    assert(key < p_tensor_basis->size(p_tensor_basis->depth()));
    std::lock_guard<std::recursive_mutex> access(m_bracketing_lock);
    auto &found = m_bracketing_cache[key];
    if (found) {
        return found;
    }

    found.set();
    if (key == 0) {
    } else if (key <= width) {
        found.reserve(1);
        found.emplace_back(key, 1);
        found.shrink_to_fit();
    } else {
        lie_product_operator op(*p_lie_basis);
        auto kl = p_tensor_basis->lparent(key);
        auto kr = p_tensor_basis->rparent(key);
        static_cast<std::vector<std::pair<key_type, int>>&>(found)
            = op(rbracket(kl), rbracket(kr));
    }
    return found;
}
