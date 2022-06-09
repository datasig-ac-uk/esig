//
// Created by sam on 23/03/2022.
//

#ifndef ESIG_PATHS_SPARSE_MUTABLE_REFERENCE_H
#define ESIG_PATHS_SPARSE_MUTABLE_REFERENCE_H

#include "esig/implementation_types.h"
#include <esig/algebra/coefficients.h>
#include <utility>

namespace esig {
namespace algebra {
namespace dtl {

template <typename Map, typename Scalar>
class sparse_mutable_reference
{
    Map& m_data;
    key_type m_key;
    static const Scalar zero;

public:

    explicit sparse_mutable_reference(Map& map, key_type key) : m_data(map), m_key(key)
    {}

    operator const Scalar&() const noexcept
    {
        auto found = m_data.find(m_key);
        if (found != m_data.end()) {
            return found->second;
        }
        return zero;
    }

    sparse_mutable_reference& operator=(const Scalar& other)
    {
        if (other != Scalar(0)) {
            m_data[m_key] = other;
        }
        return *this;
    }
    sparse_mutable_reference& operator=(Scalar&& other) noexcept
    {
        if (other != Scalar(0)) {
            m_data[m_key] = std::move(other);
        }
    }

    sparse_mutable_reference& operator+=(Scalar other)
    {
        auto found = m_data.find(m_key);
        auto new_val = other;
        if (found != m_data.end()) {
            new_val += found->second;
        }
        if (new_val != Scalar(0)) {
            m_data[m_key] = new_val;
        }
        return *this;
    }

    sparse_mutable_reference& operator-=(Scalar other)
    {
        auto found = m_data.find(m_key);
        auto new_val = -other;
        if (found != m_data.end()) {
            new_val += found->second;
        }
        if (new_val != Scalar(0)) {
            m_data[m_key] = new_val;
        }
        return *this;
    }
};

template <typename Map, typename Scalar>
const Scalar sparse_mutable_reference<Map, Scalar>::zero(0);

template<typename Map, typename S>
struct underlying_ctype<sparse_mutable_reference<Map, S>> {
    using type = S;
};



template <typename M, typename S>
struct coefficient_type_trait<sparse_mutable_reference<M, S>>
{
    using value_type = S;
    using reference = sparse_mutable_reference<M, S>;
    using const_reference = const S&;

    using value_wrapper = coefficient_implementation<S>;
    using reference_wrapper = coefficient_implementation<reference>;
    using const_reference_wrapper = coefficient_implementation<const_reference>;

    static coefficient make(sparse_mutable_reference<M, S> arg)
    {
        return coefficient(
                std::shared_ptr<coefficient_interface>(new reference_wrapper(std::move(arg)))
        );
    }
};





template <typename Map, typename Scalar>
class ESIG_ALGEBRA_EXPORT coefficient_implementation<sparse_mutable_reference<Map, Scalar>> : public coefficient_interface
{
    using proxied_type = sparse_mutable_reference<Map, Scalar>;

    proxied_type m_data;

public:
    explicit coefficient_implementation(proxied_type&& arg);
//    explicit coefficient_implementation(proxied_type arg);

    coefficient_type ctype() const noexcept override;
    bool is_const() const noexcept override;
    bool is_val() const noexcept override;
    scalar_t as_scalar() const override;
    void assign(coefficient val) override;
    coefficient add(const coefficient_interface &other) const override;
    coefficient sub(const coefficient_interface &other) const override;
    coefficient mul(const coefficient_interface &other) const override;
    coefficient div(const coefficient_interface &other) const override;
};

template<typename Map, typename Scalar>
coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::coefficient_implementation(sparse_mutable_reference<Map, Scalar>&& arg)
     : m_data(std::move(arg))
{
}


template<typename Map, typename Scalar>
coefficient_type coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::ctype() const noexcept
{
    return dtl::get_coeff_type(Scalar(0));
}
template<typename Map, typename Scalar>
bool coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::is_const() const noexcept
{
    return false;
}
template<typename Map, typename Scalar>
bool coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::is_val() const noexcept
{
    return false;
}
template<typename Map, typename Scalar>
scalar_t coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::as_scalar() const
{
    return static_cast<scalar_t>(static_cast<const Scalar&>(m_data));
}
template<typename Map, typename Scalar>
void coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::assign(coefficient val)
{
    auto& iface = *val.p_impl.get();
    auto& vtype = typeid(iface);
    if (vtype == typeid(*this)) {
        const auto& odata = dynamic_cast<const coefficient_implementation&>(*val.p_impl).m_data;
        m_data = static_cast<const Scalar&>(odata);
    } else if (vtype == typeid(coefficient_implementation<Scalar>&)) {
        const auto& other = dynamic_cast<const coefficient_implementation<Scalar>&>(*val.p_impl);
        m_data = dtl::coefficient_value_helper::value(other);
    } else if (vtype == typeid(coefficient_implementation<const Scalar&>&)) {
        const auto& other = dynamic_cast<const coefficient_implementation<const Scalar&>&>(*val.p_impl);
        m_data = dtl::coefficient_value_helper::value(other);
    } else if (vtype == typeid(coefficient_implementation<Scalar&>&)) {
        const auto& other = dynamic_cast<const coefficient_implementation<Scalar&>&>(*val.p_impl);
        m_data = dtl::coefficient_value_helper::value(other);
    } else {
        throw std::bad_cast();
    }
}
template<typename Map, typename Scalar>
coefficient coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::add(const coefficient_interface &other) const
{
    const auto& o_data = dynamic_cast<const coefficient_implementation&>(other).m_data;
    return coefficient(static_cast<const Scalar&>(m_data) + static_cast<const Scalar&>(o_data));
}
template<typename Map, typename Scalar>
coefficient coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::sub(const coefficient_interface &other) const
{
    const auto &o_data = dynamic_cast<const coefficient_implementation &>(other).m_data;
    return coefficient(static_cast<const Scalar &>(m_data) - static_cast<const Scalar &>(o_data));
}
template<typename Map, typename Scalar>
coefficient coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::mul(const coefficient_interface &other) const
{
    const auto &o_data = dynamic_cast<const coefficient_implementation &>(other).m_data;
    return coefficient(static_cast<const Scalar &>(m_data) * static_cast<const Scalar &>(o_data));
}
template<typename Map, typename Scalar>
coefficient coefficient_implementation<sparse_mutable_reference<Map, Scalar>>::div(const coefficient_interface &other) const
{
    const auto &o_data = dynamic_cast<const coefficient_implementation &>(other).m_data;
    return coefficient(static_cast<const Scalar &>(m_data) / static_cast<const Scalar &>(o_data));
}


} // namespace dtl
} // namespace algebra
} // namespace esig

#endif//ESIG_PATHS_SPARSE_MUTABLE_REFERENCE_H
