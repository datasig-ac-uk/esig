//
// Created by user on 09/09/22.
//



#include "basis_interface.h"
#include "esig/algebra/base.h"

namespace esig {
namespace algebra {

boost::optional<deg_t> BasisInterface::width() const noexcept {
    return {};
}
boost::optional<deg_t> BasisInterface::depth() const noexcept {
    return {};
}
boost::optional<deg_t> BasisInterface::degree(key_type key) const noexcept {
    return {};
}
std::string BasisInterface::key_to_string(key_type key) const noexcept {
    return std::to_string(key);
}
dimn_t BasisInterface::size(int) const noexcept {
    return 0;
}
dimn_t BasisInterface::start_of_degree(int) const noexcept {
    return 0;
}
boost::optional<key_type> BasisInterface::lparent(key_type key) const noexcept {
    return {};
}
boost::optional<key_type> BasisInterface::rparent(key_type key) const noexcept {
    return {};
}
key_type BasisInterface::index_to_key(dimn_t idx) const noexcept {
    return key_type(idx);
}
dimn_t BasisInterface::key_to_index(key_type key) const noexcept {
    return dimn_t(key);
}
let_t BasisInterface::first_letter(key_type key) const noexcept {
    return let_t(key);
}
key_type BasisInterface::key_of_letter(let_t letter) const noexcept {
    return key_type(letter);
}
bool BasisInterface::letter(key_type key) const noexcept {
    return true;
}
}// namespace algebra
}// namespace esig
