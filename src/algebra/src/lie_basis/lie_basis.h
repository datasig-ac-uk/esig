//
// Created by user on 07/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_NON_TEMPLATED_IMPL_LIE_BASIS_H_
#define ESIG_PATHS_SRC_ALGEBRA_NON_TEMPLATED_IMPL_LIE_BASIS_H_

#include <esig/implementation_types.h>
#include <esig/algebra/basis.h>
#include "hall_set.h"
#include "hall_extension.h"

#include <mutex>
#include <memory>
#include <unordered_map>

namespace esig {
namespace algebra {

inline static deg_t degree_let_func(let_t let) noexcept
{
    return 1;
}
inline static deg_t degree_binop_func(deg_t a, deg_t b) noexcept
{
    return a + b;
}

std::string key_to_string_let_func(let_t let);
std::string key_to_string_binop(const std::string&, const std::string&);
namespace dtl {
struct parent_hash {
    static_assert(sizeof(std::size_t) >= 2 * sizeof(let_t), "pair hash undefined if sizeof(size_t) < 2*sizeof(let_t)");

    inline std::size_t operator()(const std::pair<let_t, let_t> &parents) const
    {
        return (static_cast<std::size_t>(parents.first) << 8 * sizeof(let_t)) + static_cast<std::size_t>(parents.second);
    }
};

} // namespace dtl


class lie_basis : public algebra_basis
{
    std::shared_ptr<hall_set> m_hall_set;
    deg_t m_width;
    deg_t m_depth;

    hall_extension<deg_t (*)(let_t), deg_t (*)(deg_t, deg_t)> m_degree_impl;
    hall_extension<std::string (*)(let_t), std::string (*)(const std::string&, const std::string&)> m_k2s_impl;

    mutable std::recursive_mutex prod_lock;
    mutable std::unordered_map<typename hall_set::parent_type, std::vector<std::pair<key_type, int>>, dtl::parent_hash> prod_cache;

public:

    explicit lie_basis(deg_t, deg_t);

    deg_t width() const noexcept override;
    deg_t depth() const noexcept override;

    std::shared_ptr<hall_set> get_hall_set() const noexcept;

    deg_t degree(const key_type &type) const override;
    std::string key_to_string(const key_type &type) const override;
    key_type key_of_letter(let_t let) const noexcept override;
    bool letter(const key_type &type) const noexcept override;
    dimn_t size(int degree) const noexcept override;
    dimn_t start_of_degree(deg_t deg) const noexcept override;
    key_type lparent(const key_type &type) const noexcept override;
    key_type rparent(const key_type &type) const noexcept override;
    let_t first_letter(const key_type &type) const noexcept override;

private:
    std::vector<std::pair<key_type, int>>
    prod_impl(const key_type&, const key_type&) const;

public:

    const std::vector<std::pair<key_type, int>>&
    prod(const key_type&, const key_type&) const;

};

} // namespace algebra
} // namespace paths-old

#endif//ESIG_PATHS_SRC_ALGEBRA_NON_TEMPLATED_IMPL_LIE_BASIS_H_
