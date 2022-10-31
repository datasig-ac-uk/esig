//
// Created by user on 09/09/22.
//



#include "basis_interface.h"
#include "esig/algebra/base.h"

namespace esig {
namespace algebra {

boost::optional<deg_t> basis_interface::width() const noexcept {
    return {};
}
boost::optional<deg_t> basis_interface::depth() const noexcept {
    return {};
}
boost::optional<deg_t> basis_interface::degree(key_type key) const noexcept {
    return {};
}
std::string basis_interface::key_to_string(key_type key) const noexcept {
    return std::to_string(key);
}
dimn_t basis_interface::size(int) const noexcept {
    return 0;
}
dimn_t basis_interface::start_of_degree(int) const noexcept {
    return 0;
}
boost::optional<key_type> basis_interface::lparent(key_type key) const noexcept {
    return {};
}
boost::optional<key_type> basis_interface::rparent(key_type key) const noexcept {
    return {};
}
key_type basis_interface::index_to_key(dimn_t idx) const noexcept {
    return key_type(idx);
}
dimn_t basis_interface::key_to_index(key_type key) const noexcept {
    return dimn_t(key);
}
let_t basis_interface::first_letter(key_type key) const noexcept {
    return let_t(key);
}
key_type basis_interface::key_of_letter(let_t letter) const noexcept {
    return key_type(letter);
}
bool basis_interface::letter(key_type key) const noexcept {
    return true;
}
}// namespace algebra
}// namespace esig
