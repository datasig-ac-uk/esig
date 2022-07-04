//
// Created by user on 07/06/22.
//

#include <esig/algebra/python_interface.h>
#include "py_keys.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace esig {
namespace algebra {

py_lie_key::py_lie_key(const context* ctx, key_type key)
    : m_key(key), m_ctx(ctx)
{}
py_lie_key::operator key_type() const noexcept
{
    return m_key;
}
const context *py_lie_key::get_context() const noexcept
{
    return m_ctx;
}
std::string py_lie_key::to_string() const
{
    const auto& lbasis = m_ctx->borrow_lbasis();
    return lbasis.key_to_string(m_key);
}

py_lie_key py_lie_key::lparent() const
{
    const auto& lbasis = m_ctx->borrow_lbasis();
    return py_lie_key(m_ctx, lbasis.lparent(m_key));
}
py_lie_key py_lie_key::rparent() const
{
    const auto& lbasis = m_ctx->borrow_lbasis();
    return py_lie_key(m_ctx, lbasis.rparent(m_key));
}
deg_t py_lie_key::degree() const
{
    const auto& lbasis = m_ctx->borrow_lbasis();
    return lbasis.degree(m_key);
}

bool py_lie_key::equals(const py_lie_key &other) const noexcept
{
    return m_key == other.m_key;
}
bool py_lie_key::less(const py_lie_key &other) const noexcept
{
    return m_key < other.m_key;
}


lie py_lie_key::to_lie(const coefficient& c) const
{
    std::pair<key_type, coefficient> tmp(m_key, c);

    vector_construction_data data(&tmp, &tmp + 1, vector_type::sparse);
    return m_ctx->construct_lie(data);
}

} // namespace algebra
} // namespace esig

void esig::algebra::init_py_lie_key(pybind11::module_& m)
{
    using esig::algebra::py_lie_key;
    py::class_<py_lie_key> klass(m, "LieKey");

    klass.def("__str__", &py_lie_key::to_string);

}
