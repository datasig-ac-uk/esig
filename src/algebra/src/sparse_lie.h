//
// Created by user on 21/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_TMP_SRC_SPARSE_LIE_H_
#define ESIG_PATHS_SRC_ALGEBRA_TMP_SRC_SPARSE_LIE_H_

#include <esig/implementation_types.h>
#include "lie_basis/lie_basis.h"
#include <esig/algebra/iteration.h>
#include <esig/algebra/algebra_traits.h>

#include "sparse_kv_iterator.h"
#include "sparse_mutable_reference.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>


namespace esig {
namespace algebra {

template<typename Scalar>
class sparse_lie
{
    using map_type = std::map<key_type, Scalar>;
    map_type m_data;
    std::shared_ptr<lie_basis> m_basis;

    static const Scalar zero;

    void check_compatible(const lie_basis &other) const
    {
        if (m_basis->width() != other.width()) {
            throw std::invalid_argument("mismatched width");
        }
    }

    void swap(sparse_lie &other) noexcept
    {
        std::swap(m_data, other.m_data);
        std::swap(m_basis, other.m_basis);
    }


public:
    using scalar_type = Scalar;
    using basis_type = lie_basis;
    using const_iterator = typename map_type::const_iterator;

    explicit sparse_lie(deg_t width, deg_t depth)
        : m_basis(new lie_basis(width, depth)), m_data()
    {}

    explicit sparse_lie(std::shared_ptr<lie_basis> basis)
        : m_basis(std::move(basis)), m_data()
    {}

    sparse_lie(std::shared_ptr<lie_basis> basis, std::initializer_list<std::pair<const key_type, Scalar>> args)
        : m_basis(std::move(basis)), m_data(args)
    {}

    sparse_lie(std::shared_ptr<lie_basis> basis, key_type key, Scalar s)
        : m_basis(std::move(basis)), m_data{{key, s}}
    {}

    sparse_lie(std::shared_ptr<lie_basis> basis, const Scalar *begin, const Scalar *end)
        : m_basis(std::move(basis))
    {
        key_type k = 1;
        for (auto p = begin; p != end; ++p) {
            m_data[k++] = *p;
        }
    }


    template <typename InputIt>
    sparse_lie(std::shared_ptr<lie_basis> basis, InputIt begin, InputIt end)
        : m_basis(std::move(basis)), m_data()
    {
        for (auto it = begin; it != end; ++it) {
            if (it->second != zero) {
                m_data[it->first] = it->second;
            }
        }
    }

public:
    dimn_t size() const
    {
        return m_data.size();
    }
    deg_t degree() const
    {
        auto last = std::rbegin(m_data);
        if (last != std::rend(m_data)) {
            return m_basis->degree(last->first);
        }
        return 0;
    }
    deg_t width() const noexcept
    {
        return m_basis->width();
    }
    deg_t depth() const noexcept
    {
        return m_basis->depth();
    }
    vector_type storage_type() const noexcept
    {
        return vector_type::sparse;
    }
    coefficient_type coeff_type() const noexcept
    {
        return dtl::get_coeff_type(zero);
    }

    void clear() noexcept
    {
        m_data.clear();
    }
    const Scalar &operator[](const key_type &key) const
    {
        auto found = m_data.find(key);
        if (found != m_data.end()) {
            return found->second;
        }
        return zero;
    }

    dtl::sparse_mutable_reference<map_type, Scalar> operator[](const key_type &key)
    {
        return dtl::sparse_mutable_reference<map_type, Scalar>(m_data, key);
    }

    const_iterator begin() const noexcept
    {
        return m_data.begin();
    }
    const_iterator end() const noexcept
    {
        return m_data.end();
    }

    const_iterator lower_bound(key_type k) const noexcept
    {
        return m_data.lower_bound(k);
    }

private:
    template<typename F>
    sparse_lie binary_operator_impl(const sparse_lie &other, F fn) const
    {
        check_compatible(*other.m_basis);

        sparse_lie result(m_basis);
        auto lit = m_data.begin();
        auto lend = m_data.end();
        auto rit = other.m_data.begin();
        auto rend = other.m_data.end();

        while (lit != lend && rit != rend) {
            if (lit->first < rit->first) {
                auto tmp = fn(lit->second, zero);
                if (tmp != zero) {
                    result.m_data[lit->first] = std::move(tmp);
                }
                ++lit;
            } else if (rit->first < lit->first) {
                auto tmp = fn(zero, rit->second);
                if (tmp != zero) {
                    result.m_data[rit->first] = std::move(tmp);
                }
                ++rit;
            } else {
                auto tmp = fn(lit->second, rit->second);
                if (tmp != Scalar(0)) {
                    result.m_data[lit->first] = std::move(tmp);
                }
                ++lit;
                ++rit;
            }
        }

        for (; lit != lend; ++lit) {
            result.m_data[lit->first] = fn(lit->second, Scalar(0));
        }
        for (; rit != rend; ++rit) {
            result.m_data[rit->first] = fn(Scalar(0), rit->second);
        }

        return result;
    }

public:
    sparse_lie operator+(const sparse_lie &other) const
    {
        return binary_operator_impl(other, [](const Scalar &a, const Scalar &b) { return a + b; });
    }
    sparse_lie operator-(const sparse_lie &other) const
    {
        return binary_operator_impl(other, [](const Scalar &a, const Scalar &b) { return a - b; });
    }


private:
    template<typename F>
    sparse_lie unary_operator_impl(F fn) const
    {
        sparse_lie result(m_basis);
        for (const auto &item : m_data) {
            result.m_data[item.first] = fn(item.second);
        }
        return result;
    }

public:
    sparse_lie operator-() const
    {
        return unary_operator_impl([](const Scalar &a) { return -a; });
    }
    sparse_lie operator*(Scalar other) const
    {
        if (other == zero) {
            return sparse_lie(m_basis);
        }
        return unary_operator_impl([=](const Scalar &a) { return a * other; });
    }
    sparse_lie operator/(Scalar other) const
    {
        return unary_operator_impl([=](const Scalar &a) { return a / other; });
    }

private:
    template<typename F>
    void inplace_binary_operation_impl(const sparse_lie &other, F fn)
    {
        auto tmp = binary_operator_impl(other, fn);
        this->swap(tmp);
    }

public:
    sparse_lie &operator+=(const sparse_lie &other)
    {
        inplace_binary_operation_impl(other, [](const Scalar &a, const Scalar &b) { return a + b; });
        return *this;
    }
    sparse_lie &operator-=(const sparse_lie &other)
    {
        inplace_binary_operation_impl(other, [](const Scalar &a, const Scalar &b) { return a - b; });
        return *this;
    }

    sparse_lie &operator*=(Scalar other)
    {
        if (other == zero) {
            m_data.clear();
        } else {
            for (auto &val : m_data) {
                val.second *= other;
            }
        }
        return *this;
    }
    sparse_lie &operator/=(Scalar other)
    {
        for (auto &val : m_data) {
            val.second /= other;
        }
        return *this;
    }


    sparse_lie &add_scal_prod(const sparse_lie &other, Scalar scal)
    {
        if (scal != zero) {
            inplace_binary_operation_impl(other, [=](const Scalar &a, const Scalar &b) { return a + b * scal; });
        }
        return *this;
    }
    sparse_lie &sub_scal_prod(const sparse_lie &other, Scalar scal)
    {
        if (scal != zero) {
            inplace_binary_operation_impl(other, [=](const Scalar &a, const Scalar &b) { return a - b * scal; });
        }
        return *this;
    }
    sparse_lie &add_scal_div(const sparse_lie &other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](const Scalar &a, const Scalar &b) { return a + b / scal; });
        return *this;
    }
    sparse_lie &sub_scal_div(const sparse_lie &other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](const Scalar &a, const Scalar &b) { return a - b / scal; });
        return *this;
    }


    sparse_lie& add_scal_prod(key_type key, Scalar scal)
    {
        operator[](key) += scal;
        return *this;
    }
    sparse_lie& sub_scal_prod(key_type key, Scalar scal)
    {
        operator[](key) -= scal;
        return *this;
    }


private:
    template<typename Op>
    static void mul_impl(
            sparse_lie &out_lie,
            const sparse_lie &lhs,
            const sparse_lie &rhs,
            Op op,
            deg_t max_degree)
    {
        std::map<key_type, Scalar> tmp;

        for (int out_deg = 0; out_deg <= max_degree; ++out_deg) {
            for (int lhs_deg = out_deg; lhs_deg >= 0; --lhs_deg) {
                auto rhs_deg = out_deg - lhs_deg;
                auto lbegin = lhs.m_data.lower_bound(lhs.m_basis->start_of_degree(lhs_deg));
                auto lend = lhs.m_data.lower_bound(lhs.m_basis->start_of_degree(lhs_deg + 1));
                auto rbegin = rhs.m_data.lower_bound(rhs.m_basis->start_of_degree(rhs_deg));
                auto rend = rhs.m_data.lower_bound(rhs.m_basis->start_of_degree(rhs_deg + 1));

                for (auto lit=lbegin; lit != lend; ++lit) {
                    for (auto rit=rbegin; rit != rend; ++rit) {
                        for (const auto &v : out_lie.m_basis->prod(lit->first, rit->first)) {
                            tmp[v.first] += op(v.second * (lit->second) * (rit->second));
                        }
                    }
                }
            }
        }

        auto it = out_lie.end();
        for (const auto &val : tmp) {
            if (val.second != zero) {
                it = out_lie.m_data.emplace_hint(it, val.first, val.second);
            }
        }
    }


public:
    sparse_lie operator*(const sparse_lie &other) const
    {
        check_compatible(*other.m_basis);

        sparse_lie tmp(m_basis);
        mul_impl(
                tmp, *this, other, [](Scalar s) { return s; }, m_basis->depth());
        return tmp;
    }
    sparse_lie &operator*=(const sparse_lie &other)
    {
        check_compatible(*other.m_basis);

        sparse_lie tmp(m_basis);
        mul_impl(
                tmp, *this, other, [](Scalar s) { return s; }, m_basis->depth());
        this->swap(tmp);
        return *this;
    }
    sparse_lie &mul_scal_prod(const sparse_lie &other, Scalar s, deg_t max_depth)
    {
        // Assume compatible
        sparse_lie tmp(m_basis);
        mul_impl(
                tmp, *this, other, [=](Scalar v) { return v * s; }, max_depth);
        this->swap(tmp);
        return *this;
    }
    sparse_lie &mul_scal_prod(const sparse_lie &other, Scalar s)
    {
        sparse_lie tmp(m_basis);
        mul_impl(
                tmp, *this, other, [=](Scalar v) { return v * s; }, m_basis->depth());
        this->swap(tmp);
        return *this;
    }
    sparse_lie &mul_scal_div(const sparse_lie &other, Scalar s)
    {
        sparse_lie tmp(m_basis);
        mul_impl(
                tmp, *this, other, [=](Scalar v) { return v / s; }, m_basis->depth());
        this->swap(tmp);
        return *this;
    }
    sparse_lie &mul_scal_div(const sparse_lie &other, Scalar s, deg_t max_depth)
    {
        sparse_lie tmp(m_basis);
        mul_impl(
                tmp, *this, other, [=](Scalar v) { return v / s; }, max_depth);
        this->swap(tmp);
        return *this;
    }


    sparse_lie &add_mul(const sparse_lie &lhs, const sparse_lie &rhs)
    {
        mul_impl(*this, lhs, rhs, [](Scalar s) { return s; }, m_basis->depth());
        return *this;
    }
    sparse_lie &sub_mul(const sparse_lie &lhs, const sparse_lie &rhs)
    {
        mul_impl(*this, lhs, rhs, [](Scalar s) { return -s; }, m_basis->depth());
        return *this;
    }


    friend std::ostream &operator<<(std::ostream &os, const sparse_lie &arg)
    {
        os << "{ ";
        for (const auto &val : arg.m_data) {
            os << val.second << arg.m_basis->key_to_string(val.first) << ' ';
        }
        return os << '}';
    }

    dtl::sparse_kv_iterator<sparse_lie> iterate_kv() const
    {
        return {m_data.begin(), m_data.end()};
    }

    bool operator==(const sparse_lie &other) const
    {
        auto lit = m_data.begin();
        auto lend = m_data.end();
        auto rit = other.m_data.begin();
        auto rend = other.m_data.end();

        for (; lit != lend && rit != rend; ++lit, ++rit) {
            if (lit->first != rit->first || lit->second != rit->second) {
                return false;
            }
        }
        return lit == lend && rit == rend;
    }
};


template <typename S>
struct algebra_info<sparse_lie<S>>
{
    using scalar_type = S;

    static constexpr coefficient_type ctype() noexcept
    { return dtl::get_coeff_type(S(0)); }
    static constexpr vector_type vtype() noexcept
    { return vector_type::sparse; }
    static deg_t width(const sparse_lie<S>& instance) noexcept
    { return instance.width(); }
    static deg_t max_depth(const sparse_lie<S>& instance) noexcept
    { return instance.depth(); }
    static key_type convert_key(esig::key_type key) noexcept
    { return key; }

};

namespace dtl {

template<typename S>
class dense_data_access_implementation<sparse_lie<S>>
    : public dense_data_access_interface
{
    typename sparse_lie<S>::const_iterator m_current, m_end;
public:

    dense_data_access_implementation(const sparse_lie<S>& alg, key_type start)
        : m_current(alg.lower_bound(start)), m_end(alg.end())
    {}


    dense_data_access_item next() override {
        if (m_current != m_end) {
            auto key = m_current->first;
            const auto* p = &m_current->second;
            ++m_current;
            return {key, p, p + 1};
        }
        return {0, nullptr, nullptr};
    }
};

}

template<typename Scalar>
const Scalar sparse_lie<Scalar>::zero(0);

extern template class sparse_lie<double>;
extern template class sparse_lie<float>;

}// namespace algebra
}// namespace esig
#endif//ESIG_PATHS_SRC_ALGEBRA_TMP_SRC_SPARSE_LIE_H_