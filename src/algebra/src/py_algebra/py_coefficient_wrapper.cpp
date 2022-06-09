//
// Created by sam on 27/05/22.
//

#include <esig/algebra/python_interface.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;


namespace esig {
namespace algebra {
namespace dtl {

coefficient_implementation<py::object>::coefficient_implementation(py::object arg)
    : m_data(std::move(arg))
{
}
bool coefficient_implementation<py::object>::is_const() const noexcept
{
    return false;
}
bool coefficient_implementation<py::object>::is_val() const noexcept
{
    return false;
}
scalar_t coefficient_implementation<py::object>::as_scalar() const
{
    return m_data.cast<scalar_t>();
}
void coefficient_implementation<py::object>::assign(coefficient val)
{
    const auto& type = py::type::of(m_data);
    try {
        auto other = dynamic_cast<coefficient_implementation&>(*val.p_impl).m_data;
        if (py::isinstance(other, type)) {
            m_data = other;
        } else {
            m_data = type(other);
        }
    } catch (std::bad_cast&) {
        switch (val.ctype()) {
            case coefficient_type::dp_real:
            case coefficient_type::sp_real: m_data = type(py::float_(static_cast<scalar_t>(val))); break;
        }
    }
}

#define ESIG_MAKE_OBJECT_BINOP(NAME, OP) \
    coefficient coefficient_implementation<py::object>:: NAME (const coefficient_interface& other) const \
    {                                    \
        if (typeid(other) == typeid(*this)) {                                                          \
            const auto& rhs = dynamic_cast<const coefficient_implementation&>(other).m_data;           \
            return coefficient(m_data OP rhs);\
        } else {                         \
        }                                \
        throw std::invalid_argument("no operation " # NAME " for argument");\
    }

ESIG_MAKE_OBJECT_BINOP(add, +)
ESIG_MAKE_OBJECT_BINOP(sub, -)
ESIG_MAKE_OBJECT_BINOP(mul, *)
ESIG_MAKE_OBJECT_BINOP(div, /)

#undef ESIG_MAKE_OBJECT_BINOP

#define ESIG_MAKE_OBJECT_BINOP(NAME, OP)  \
    coefficient coefficient_implementation<py::object>::NAME(const scalar_t &other) const \
    {                                    \
        return coefficient(m_data OP py::float_(other)); \
    }

ESIG_MAKE_OBJECT_BINOP(add, +)
ESIG_MAKE_OBJECT_BINOP(sub, -)
ESIG_MAKE_OBJECT_BINOP(mul, *)
ESIG_MAKE_OBJECT_BINOP(div, /)

#undef ESIG_MAKE_OBJECT_BINOP



} // namespace dtl
} // namespace algebra
} // namespace esig
