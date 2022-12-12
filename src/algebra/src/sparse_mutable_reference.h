//
// Created by sam on 23/03/2022.
//

#ifndef ESIG_PATHS_SPARSE_MUTABLE_REFERENCE_H
#define ESIG_PATHS_SPARSE_MUTABLE_REFERENCE_H

#include "esig/implementation_types.h"
#include <utility>

namespace esig {
namespace algebra {
namespace dtl {

template<typename Map, typename Scalar>
class sparse_mutable_reference {
    Map &m_data;
    key_type m_key;

public:
    static const Scalar zero;

    explicit sparse_mutable_reference(Map &map, key_type key) : m_data(map), m_key(key) {}

    operator const Scalar &() const noexcept {
        auto found = m_data.find(m_key);
        if (found != m_data.end()) {
            return found->second;
        }
        return zero;
    }

    sparse_mutable_reference &operator=(const Scalar &other) {
        if (other != Scalar(0)) {
            m_data[m_key] = other;
        }
        return *this;
    }
    sparse_mutable_reference &operator=(Scalar &&other) noexcept {
        if (other != Scalar(0)) {
            m_data[m_key] = std::move(other);
        }
        return *this;
    }

    sparse_mutable_reference &operator+=(Scalar other) {
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

    sparse_mutable_reference &operator-=(Scalar other) {
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

    sparse_mutable_reference &operator*=(const Scalar &other) {
        if (other == Scalar(0)) {
            m_data.erase(m_key);
            return *this;
        }
        auto found = m_data.find(m_key);
        if (found != m_data.end()) {
            auto new_val = found->second * other;
            if (new_val != Scalar(0)) {
                found->second = new_val;
            } else {
                m_data.erase(found);
            }
        }
        return *this;
    }

    sparse_mutable_reference &operator/=(const Scalar &other) {
        if (other == Scalar(0)) {
            throw std::runtime_error("division by zero");
        }
        auto found = m_data.find(m_key);
        if (found != m_data.end()) {
            found->second /= other;
        }
        return *this;
    }
};

template<typename Map, typename Scalar>
const Scalar sparse_mutable_reference<Map, Scalar>::zero(0);

}// namespace dtl
}// namespace algebra

namespace scalars {
namespace dtl {

template<typename Map, typename Scalar>
class scalar_type_trait<algebra::dtl::sparse_mutable_reference<Map, Scalar>> {
public:
    using value_type = Scalar;
    using rational_type = Scalar;
    using reference = algebra::dtl::sparse_mutable_reference<Map, Scalar>;
    using const_reference = const Scalar &;

    using intermediate_interface = intermediate_scalar_interface<Scalar>;
    using rational_intermediate_interface = intermediate_interface;

    using v_wrapper = scalar_implementation<Scalar>;
    using r_wrapper = scalar_implementation<reference>;
    using cr_wrapper = scalar_implementation<const_reference>;

    static const scalar_type *get_type() noexcept { return scalar_type_holder<Scalar>::get_type(); }

    static scalar make(reference val) {
        return scalar(new r_wrapper(std::move(val)), get_type());
    }
};

template<typename Map, typename Scalar>
class scalar_implementation<algebra::dtl::sparse_mutable_reference<Map, Scalar>>
    : public intermediate_scalar_interface<Scalar> {
    algebra::dtl::sparse_mutable_reference<Map, Scalar> m_data;

    using trait = scalar_type_trait<Scalar>;

    using value_type = typename trait::value_type;
    using reference = typename trait::reference;
    using const_reference = typename trait::const_reference;

public:
    explicit scalar_implementation(algebra::dtl::sparse_mutable_reference<Map, Scalar> &&val)
        : m_data(std::move(val)) {}

    const scalar_type *type() const noexcept override {
        return trait::get_type();
    }

    bool is_const() const noexcept override { return false; }
    bool is_value() const noexcept override { return false; }
    bool is_zero() const noexcept override { return static_cast<const_reference>(m_data) == Scalar(0); }

    scalar_t as_scalar() const noexcept override {
        return static_cast<scalar_t>(static_cast<const_reference>(m_data));
    }
    scalar uminus() const noexcept override {
        return trait::make(-static_cast<const_reference>(m_data));
    }
    value_type into_value() const noexcept override { return static_cast<const_reference>(m_data); }
    const_reference into_cref() const noexcept override { return static_cast<const_reference>(m_data); }

    //
    //#define ESIG_SCALAR_GENERATE_DF(NAME, OP)                                                                               \
//    scalar NAME(const scalar &other) const override {                                                                   \
//        if (this->type() == other.type()) {                                                                             \
//            const auto &o_ref = static_cast<const typename trait::intermediate_interface *>(other.p_data)->into_cref(); \
//            return trait::make(m_data OP o_ref);                                                                        \
//        }                                                                                                               \
//        return trait::make(m_data OP static_cast<value_type>(other.to_scalar_t()));                                     \
//    }
    //
    //    ESIG_SCALAR_GENERATE_DF(add, +)
    //    ESIG_SCALAR_GENERATE_DF(sub, -)
    //    ESIG_SCALAR_GENERATE_DF(mul, *)
    //
    //#undef ESIG_SCALAR_GENERATE_DF

    //    scalar div(const scalar &other) const override {
    //        if (this->type()->rational_type() == other.type()) {
    //            const auto &o_ref = static_cast<const typename trait::rational_intermediate_interface *>(other.p_data)->into_cref();
    //            return trait::make(m_data / o_ref);
    //        }
    //        return trait::make(m_data / static_cast<value_type>(other.to_scalar_t()));
    //    }

#define ESIG_SCALAR_GENERATE_DFI(NAME, OP)                                                   \
    void NAME(const scalar &other) override {                                                \
        if (this->type() == other.type()) {                                                  \
            const auto &o_ref = *other.to_const_pointer().template raw_cast<const Scalar>(); \
            m_data OP o_ref;                                                                 \
        } else {                                                                             \
            m_data OP static_cast<value_type>(other.to_scalar_t());                          \
        }                                                                                    \
    }

    ESIG_SCALAR_GENERATE_DFI(assign, =)
    ESIG_SCALAR_GENERATE_DFI(add_inplace, +=)
    ESIG_SCALAR_GENERATE_DFI(sub_inplace, -=)
    ESIG_SCALAR_GENERATE_DFI(mul_inplace, *=)

#undef ESIG_SCALAR_GENERATE_DFI

    void div_inplace(const scalar &other) override {
        if (this->type()->rational_type() == other.type()) {
            const auto &o_ref = *other.to_const_pointer().template raw_cast<const Scalar>();
            m_data /= o_ref;
        } else {
            m_data /= static_cast<typename trait::rational_type>(other.to_scalar_t());
        }
    }

    void assign(scalar_pointer pointer) override {
        auto *this_type = type();
        if (pointer.type() == this_type) {
            m_data = *pointer.raw_cast<const value_type>();
        } else {
            auto tmp = this_type->convert(pointer);
            m_data = *tmp.to_const_pointer().template raw_cast<const value_type>();
        }
    }
    scalar_pointer to_pointer() override {
        throw std::runtime_error("cannot retrieve mutable pointer");
    }
    scalar_pointer to_pointer() const noexcept override {
        return {&static_cast<const_reference>(m_data), type()};
    }
};

}// namespace dtl
}// namespace scalars
}// namespace esig

#endif//ESIG_PATHS_SPARSE_MUTABLE_REFERENCE_H
