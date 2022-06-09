//
// Created by sam on 06/06/22.
//

#include "py_free_tensor.h"
#include "esig/algebra/python_interface.h"


namespace esig {
namespace algebra {


py_tensor_key::py_tensor_key(const context* ctx, key_type key)
    :  m_ctx(ctx), m_key(key)
{}
py_tensor_key::operator key_type() const noexcept
{
    return m_key;
}
const context *py_tensor_key::get_context() const noexcept
{
    return m_ctx;
}
std::string py_tensor_key::to_string() const
{
    const auto& tbasis = m_ctx->borrow_tbasis();
    return tbasis.key_to_string(m_key);
}
py_tensor_key py_tensor_key::lparent() const
{
    const auto& tbasis = m_ctx->borrow_tbasis();
    return py_tensor_key {m_ctx, tbasis.lparent(m_key)};
}
py_tensor_key py_tensor_key::rparent() const
{
    const auto& tbasis = m_ctx->borrow_tbasis();
    return py_tensor_key {m_ctx, tbasis.rparent(m_key)};
}
deg_t py_tensor_key::degree() const
{
    const auto& tbasis = m_ctx->borrow_tbasis();
    return tbasis.degree(m_key);
}
bool py_tensor_key::equals(const py_tensor_key &other) const noexcept
{
    return m_key == other.m_key;
}
bool py_tensor_key::less(const py_tensor_key &other) const noexcept
{
    return m_key < other.m_key;
}

} // namespace algebra
} // namespace esig
