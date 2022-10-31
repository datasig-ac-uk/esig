//
// Created by user on 09/09/22.
//


#include "basis_interface.h"
#include "esig/algebra/base.h"

using namespace esig;
using namespace algebra;

boost::optional<deg_t> esig::algebra::basis::width() const noexcept
{
    return p_impl->width();
}
boost::optional<deg_t> basis::depth() const noexcept {
    return p_impl->depth();
}
boost::optional<deg_t> basis::degree(key_type key) const noexcept {
    return p_impl->degree(key);
}
std::string basis::key_to_string(key_type key) const noexcept {
    p_impl->key_to_string(key);
}
dimn_t basis::size(int deg) const noexcept {
    return p_impl->size(deg);
}
dimn_t basis::start_of_degree(int deg) const noexcept {
    return p_impl->start_of_degree(deg);
}
boost::optional<key_type> basis::lparent(key_type key) const noexcept {
    return p_impl->lparent(key);
}
boost::optional<key_type> basis::rparent(key_type key) const noexcept {
    return p_impl->rparent(key);
}
key_type basis::index_to_key(dimn_t idx) const noexcept {
    return p_impl->index_to_key(idx);
}
dimn_t basis::key_to_index(key_type key) const noexcept {
    return p_impl->key_to_index(key);
}
let_t basis::first_letter(key_type key) const noexcept {
    return p_impl->first_letter(key);
}
key_type basis::key_of_letter(let_t letter) const noexcept {
    return p_impl->key_of_letter(letter);
}
bool basis::letter(key_type key) const noexcept {
    return p_impl->letter(key);
}
