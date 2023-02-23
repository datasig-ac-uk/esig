//
// Created by sam on 13/12/22.
//

#include "esig/scalars.h"


using namespace esig;
using namespace esig::scalars;


key_scalar_array::~key_scalar_array()
{
    if (p_type != nullptr && m_scalars_owned) {
        auto size = scalar_array::size();
        const auto* tp = scalar_array::type();
        tp->deallocate(*this, size);
    }
    if (m_keys_owned ){
        delete[] p_keys;
    }
}

key_scalar_array::key_scalar_array(const key_scalar_array &other)
    : scalar_array(other.type()->allocate(other.size()), other.size()),
      m_scalars_owned(true)
{
    if (other.p_data != nullptr) {
        p_type->convert_copy(const_cast<void*>(p_data), other, m_size);

        if (p_keys != nullptr) {
            allocate_keys();
            std::copy(other.p_keys, other.p_keys + other.m_size, const_cast<key_type*>(p_keys));
        }
    } else {
        assert(other.p_keys == nullptr);
    }
}
key_scalar_array::key_scalar_array(key_scalar_array &&other) noexcept
    : scalar_array(other),
      p_keys(other.p_keys),
      m_scalars_owned(other.m_scalars_owned),
      m_keys_owned(other.m_keys_owned)
{
    other.m_scalars_owned = false;
    other.p_keys = nullptr;
    other.p_data = nullptr;
    assert(other.p_data == nullptr);
}
key_scalar_array::key_scalar_array(owned_scalar_array &&sa) noexcept
    : scalar_array(std::move(sa)), m_scalars_owned(true)
{
}
key_scalar_array::key_scalar_array(scalar_array base, const key_type *keys)
    : scalar_array(base), p_keys(keys), m_keys_owned(false)
{
}
key_scalar_array::key_scalar_array(const scalar_type *type) noexcept : scalar_array(type) {
}
key_scalar_array::key_scalar_array(const scalar_type *type, dimn_t n) noexcept
    : scalar_array(type)
{
    allocate_scalars(static_cast<idimn_t>(n));
}

key_scalar_array::key_scalar_array(const scalar_type *type, const void *begin, dimn_t count) noexcept
    : scalar_array(begin, type, count), m_scalars_owned(false)
{
}
key_scalar_array &key_scalar_array::operator=(const scalar_array &other) noexcept {
    if (&other != this) {
        this->~key_scalar_array();
        scalar_array::operator=(other);
        m_scalars_owned = false;
        m_keys_owned = false;
    }
    return *this;
}
key_scalar_array &key_scalar_array::operator=(key_scalar_array &&other) noexcept {
    if (&other != this) {
        m_scalars_owned = other.m_scalars_owned;
        m_keys_owned = other.m_keys_owned;
        p_keys = other.p_keys;
        scalar_array::operator=(std::move(other));
        other.p_keys = nullptr;
    }
    return *this;
}
key_scalar_array &key_scalar_array::operator=(owned_scalar_array &&other) noexcept {
    m_scalars_owned = true;
    scalar_array::operator=(std::move(other));
    return *this;
}
void key_scalar_array::allocate_scalars(idimn_t count)
{
    auto new_size = (count == -1) ? m_size : static_cast<dimn_t>(count);
    if (new_size != 0) {
        scalar_array::operator=({p_type->allocate(new_size), new_size});
        m_scalars_owned = true;
    }
}
void key_scalar_array::allocate_keys(idimn_t count) {
    auto new_size = (count == -1) ? m_size : static_cast<dimn_t>(count);
    if (new_size != 0) {
        p_keys = new key_type[new_size];
    } else {
        p_keys = nullptr;
    }
}
key_type *key_scalar_array::keys() {
    if (m_keys_owned) {
        return const_cast<key_type*>(p_keys);
    }
    throw std::runtime_error("borrowed keys are not mutable");
}

key_scalar_array::operator owned_scalar_array() &&noexcept {
    auto result = owned_scalar_array(p_type, p_data, m_size);
    p_type = nullptr;
    p_data = nullptr;
    m_size = 0;
    return result;
}
