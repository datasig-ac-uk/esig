//
// Created by sam on 07/03/2022.
//

#ifndef ESIG_PATHS_TENSOR_BASIS_H
#define ESIG_PATHS_TENSOR_BASIS_H

#include <esig/implementation_types.h>
#include <esig/algebra/basis.h>

#include <memory>
#include <vector>

namespace esig {
namespace algebra {

class tensor_basis : public algebra_basis
{
    deg_t m_width;
    deg_t m_depth;
    std::shared_ptr<std::vector<dimn_t>> m_powers;
public:

    tensor_basis(deg_t width, deg_t depth);

    deg_t width() const noexcept override;
    deg_t depth() const noexcept override;
    deg_t degree(const key_type &type) const override;
    std::string key_to_string(const key_type &type) const override;
    key_type key_of_letter(let_t let) const noexcept override;
    bool letter(const key_type &type) const noexcept override;
    dimn_t size(int i) const noexcept override;
    dimn_t start_of_degree(deg_t deg) const noexcept override;
    key_type lparent(const key_type &type) const noexcept override;
    key_type rparent(const key_type &type) const noexcept override;
    let_t first_letter(const key_type &type) const noexcept override;

    const std::vector<dimn_t>& powers() const noexcept;

};


} // namespace algebra
} // namespace esig_paths


#endif//ESIG_PATHS_TENSOR_BASIS_H