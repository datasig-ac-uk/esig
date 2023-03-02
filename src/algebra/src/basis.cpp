//
// Created by user on 24/03/2022.
//
#include "esig/algebra/basis.h"



using namespace esig;
using namespace algebra;

boost::optional<deg_t> esig::algebra::Basis::width() const noexcept {
    return p_impl->width();
}
boost::optional<deg_t> Basis::depth() const noexcept {
    return p_impl->depth();
}
boost::optional<deg_t> Basis::degree(key_type key) const noexcept {
    return p_impl->degree(key);
}
std::string Basis::key_to_string(key_type key) const noexcept {
    return p_impl->key_to_string(key);
}
dimn_t Basis::size(int deg) const noexcept {
    return p_impl->size(deg);
}
dimn_t Basis::start_of_degree(int deg) const noexcept {
    return p_impl->start_of_degree(deg);
}
boost::optional<key_type> Basis::lparent(key_type key) const noexcept {
    return p_impl->lparent(key);
}
boost::optional<key_type> Basis::rparent(key_type key) const noexcept {
    return p_impl->rparent(key);
}
key_type Basis::index_to_key(dimn_t idx) const noexcept {
    return p_impl->index_to_key(idx);
}
dimn_t Basis::key_to_index(key_type key) const noexcept {
    return p_impl->key_to_index(key);
}
let_t Basis::first_letter(key_type key) const noexcept {
    return p_impl->first_letter(key);
}
key_type Basis::key_of_letter(let_t letter) const noexcept {
    return p_impl->key_of_letter(letter);
}
bool Basis::letter(key_type key) const noexcept {
    return p_impl->letter(key);
}
