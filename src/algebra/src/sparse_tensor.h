//
// Created by user on 21/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_TMP_SRC_SPARSE_TENSOR_H_
#define ESIG_PATHS_SRC_ALGEBRA_TMP_SRC_SPARSE_TENSOR_H_

#include <esig/implementation_types.h>
#include <esig/algebra/iteration.h>
#include <esig/algebra/algebra_traits.h>
#include "tensor_basis/tensor_basis.h"

#include "sparse_kv_iterator.h"
#include "sparse_mutable_reference.h"

#include <memory>
#include <map>
#include <iostream>
#include <algorithm>


namespace esig {
namespace algebra {

template <typename Scalar>
class sparse_tensor
{
    using map_type = std::map<key_type, Scalar>;
    map_type m_data;
    std::shared_ptr<tensor_basis> m_basis;
    static const Scalar zero;

    void check_compatible(const tensor_basis& other) const
    {
        if (m_basis->width() != other.width()) {
            throw std::invalid_argument("mismatched width");
        }
    }

    void swap(sparse_tensor& other) noexcept
    {
        std::swap(m_data, other.m_data);
        std::swap(m_basis, other.m_basis);
    }

public:

    using scalar_type = Scalar;
    using basis_type = tensor_basis;
    using const_iterator = typename map_type::const_iterator;
    static constexpr vector_type vtype = vector_type::sparse;


    explicit sparse_tensor(deg_t width, deg_t depth)
        : m_basis(new tensor_basis(width, depth)), m_data()
    {
    }

    explicit sparse_tensor(std::shared_ptr<tensor_basis> basis)
        : m_basis(std::move(basis)), m_data()
    {}

    sparse_tensor(std::shared_ptr<tensor_basis> basis, const Scalar* begin, const Scalar* end)
            : m_basis(std::move(basis))
    {
        key_type k = 0;
        for (auto p = begin; p != end; ++p) {
            m_data[k++] = *p;
        }
    }

    sparse_tensor(std::shared_ptr<tensor_basis> basis, std::initializer_list<std::pair<const key_type, Scalar>> args)
        : m_basis(std::move(basis)), m_data(args)
    {}

    sparse_tensor(std::shared_ptr<tensor_basis> basis, key_type key, Scalar s)
        : m_basis(std::move(basis)), m_data{{key, s}}
    {}

    sparse_tensor(std::shared_ptr<tensor_basis> basis, Scalar val)
            : m_basis(std::move(basis)), m_data {{0, val}}
    {}

    template <typename InputIt>
    sparse_tensor(std::shared_ptr<tensor_basis> basis, InputIt begin, InputIt end)
            : m_basis(std::move(basis)), m_data(begin, end)
    {}

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
        return dtl::get_coeff_type(Scalar(0));
    }

    const Scalar& operator[](const key_type& key) const
    {
        auto found = m_data.find(key);
        if (found != m_data.end()) {
            return found->second;
        }
        return zero;
    }

    dtl::sparse_mutable_reference<map_type, Scalar> operator[](const key_type& key)
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

private:

    template <typename F>
    sparse_tensor binary_operator_impl(const sparse_tensor& other, F fn) const
    {
        check_compatible(*other.m_basis);

        sparse_tensor result(m_basis);
        auto lit = m_data.begin();
        auto lend = m_data.end();
        auto rit = other.m_data.begin();
        auto rend = other.m_data.end();

        while (lit != lend && rit != rend) {
            if (lit->first < rit->first) {
                result.m_data[lit->first] = fn(lit->second, Scalar(0));
                ++lit;
            } else if (rit->first < lit->first) {
                result.m_data[rit->first] = fn(Scalar(0), rit->second);
                ++rit;
            } else {
                auto tmp = fn(lit->second, rit->second);
                if (tmp != Scalar(0)) {
                    result.m_data[lit->first] = tmp;
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

    sparse_tensor operator+(const sparse_tensor& other) const
    {
        return binary_operator_impl(other, [](const Scalar& a, const Scalar& b) { return a + b; });
    }
    sparse_tensor operator-(const sparse_tensor& other) const
    {
        return binary_operator_impl(other, [](const Scalar& a, const Scalar& b) { return a - b; });
    }

private:

    template <typename F>
    sparse_tensor unary_operator_impl(F fn) const
    {
        sparse_tensor result(m_basis);
        for (const auto& item : m_data) {
            result.m_data[item.first] = fn(item.second);
        }
        return result;
    }

public:

    sparse_tensor operator-() const
    {
        return unary_operator_impl([](const Scalar& a) { return -a; });
    }
    sparse_tensor operator*(Scalar other) const
    {
        return unary_operator_impl([=](const Scalar& a) { return a*other; });
    }
    sparse_tensor operator/(Scalar other) const
    {
        return unary_operator_impl([=](const Scalar& a) { return a/other; });
    }

private:

    template <typename F>
    void inplace_unary_operator(F&& fn)
    {
        for (auto& item : m_data) {
            item.second = fn(item.second);
        }
    }

public:

    sparse_tensor& operator*=(const Scalar& arg)
    {
        inplace_unary_operator([&](const Scalar& s) { return s*arg; });
        return *this;
    }
    sparse_tensor& operator/=(const Scalar& arg)
    {
        inplace_unary_operator([&](const Scalar& s) { return s/arg; });
        return *this;
    }


private:

    template <typename F>
    void inplace_binary_operation_impl(const sparse_tensor& other, F fn)
    {
        auto tmp = binary_operator_impl(other, fn);
        this->swap(tmp);
    }

public:

    sparse_tensor& operator+=(const sparse_tensor& other)
    {
        inplace_binary_operation_impl(other, [](const Scalar& a, const Scalar& b) { return a + b; });
        return *this;
    }
    sparse_tensor& operator-=(const sparse_tensor& other)
    {
        inplace_binary_operation_impl(other, [](const Scalar& a, const Scalar& b) { return a - b; });
        return *this;
    }
    sparse_tensor& add_scal_prod(const sparse_tensor& other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](const Scalar& a, const Scalar& b) { return a + b*scal; });
        return *this;
    }
    sparse_tensor& sub_scal_prod(const sparse_tensor& other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](const Scalar &a, const Scalar &b) { return a - b * scal; });
        return *this;
    }
    sparse_tensor& add_scal_div(const sparse_tensor& other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](const Scalar& a, const Scalar& b) { return a + b/scal; });
        return *this;
    }
    sparse_tensor& sub_scal_div(const sparse_tensor& other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](const Scalar &a, const Scalar &b) { return a - b / scal; });
        return *this;
    }



private:

    template <typename Op>
    inline static void inplace_mul_impl(
            sparse_tensor& out_tensor,
            const sparse_tensor& lhs,
            const sparse_tensor& rhs,
            Op op,
            deg_t max_degree
            )
    {
        std::map<key_type, Scalar> tmp;
        const auto& powers = out_tensor.m_basis->powers();

        for (int out_deg = 0; out_deg <= max_degree; ++out_deg) {
            for (int lhs_deg = out_deg; lhs_deg >= 0; --lhs_deg) {
                auto rhs_deg = out_deg - lhs_deg;
                auto llb = lhs.m_basis->start_of_degree(lhs_deg);
                auto lub = lhs.m_basis->start_of_degree(lhs_deg + 1);
                auto lbegin = lhs.m_data.lower_bound(llb);
                auto lend = lhs.m_data.lower_bound(lub);

                auto rlb = rhs.m_basis->start_of_degree(rhs_deg);
                auto rub = rhs.m_basis->start_of_degree(rhs_deg + 1);
                auto rbegin = rhs.m_data.lower_bound(rlb);
                auto rend = rhs.m_data.lower_bound(rub);

                for (auto lit = lbegin; lit != lend; ++lit) {
                    for (auto rit=rbegin; rit != rend; ++rit) {
                        tmp[lit->first * powers[rhs_deg] + rit->first] += op(lit->second * rit->second);
                    }
                }
            }
        }



        auto it = out_tensor.end();
        for (const auto& val : tmp) {
            if (val.second != Scalar(0)) {
                it = out_tensor.m_data.emplace_hint(it, val.first, val.second);
            }
        }
    }

public:


    sparse_tensor operator*(const sparse_tensor& other) const
    {
        check_compatible(*other.m_basis);
        sparse_tensor tmp(m_basis);
        inplace_mul_impl(tmp, *this, other, [](Scalar s) { return s; }, m_basis->depth());
        return tmp;
    }
    sparse_tensor& operator*=(const sparse_tensor& other)
    {
        auto tmp = operator*(other);
        this->swap(tmp);
        return *this;
    }
    sparse_tensor& mul_scal_prod(const sparse_tensor& other, Scalar s, deg_t max_depth)
    {
        sparse_tensor tmp(m_basis);
        inplace_mul_impl(tmp, *this, other, [=](Scalar v) { return v*s; }, max_depth);
        this->swap(tmp);
        return *this;
    }

    sparse_tensor& mul_scal_prod(const sparse_tensor& other, Scalar s)
    {
        return mul_scal_prod(other, s, m_basis->depth());
    }
    sparse_tensor& mul_scal_div(const sparse_tensor& other, Scalar s, deg_t max_depth)
    {
        sparse_tensor tmp(m_basis);
        inplace_mul_impl(tmp, *this, other, [=](Scalar v) { return v/s; }, max_depth);
        this->swap(tmp);
        return *this;
    }
    sparse_tensor& mul_scal_div(const sparse_tensor& other, Scalar s)
    {
        return mul_scal_div(other, s, m_basis->depth());
    }

    sparse_tensor& add_mul(const sparse_tensor& lhs, const sparse_tensor& rhs)
    {
        inplace_mul_impl(*this, lhs, rhs, [](Scalar s) { return s; }, m_basis->depth());
        return *this;
    }
    sparse_tensor& sub_mul(const sparse_tensor& lhs, const sparse_tensor& rhs)
    {
        inplace_mul_impl(*this, lhs, rhs, [](Scalar s) { return -s; }, m_basis->depth());
        return *this;
    }

    friend sparse_tensor exp(const sparse_tensor& arg)
    {
        const auto max_depth = arg.m_basis->depth();
        sparse_tensor result(arg.m_basis);
        for (deg_t d=0; d < max_depth; ++d) {
            result.mul_scal_div(arg, Scalar(max_depth - d), d+1);
            result.m_data[key_type(0)] += Scalar(1);
        }
        return result;
    }
    friend sparse_tensor log(const sparse_tensor& arg)
    {
        const auto max_depth = arg.m_basis->depth();
        sparse_tensor x(arg), result(arg.m_basis);
        auto it = x.m_data.find(key_type(0));
        if (it != x.m_data.end()) {
            x.m_data.erase(it);
        }

        for (deg_t d=0; d<max_depth; ++d) {
            auto D = max_depth - d;
            if ((D&1) == 0) {
                result.m_data[key_type(0)] -= Scalar(1) / Scalar(D);
            } else {
                result.m_data[key_type(0)] += Scalar(1) / Scalar(D);
            }
            result *= x;
        }
        return result;
    }
    friend sparse_tensor inverse(const sparse_tensor& arg)
    {
        return sparse_tensor(arg.m_basis);
    }
    sparse_tensor& fmexp_inplace(const sparse_tensor& arg)
    {
        check_compatible(*arg.m_basis);
        const auto max_depth = m_basis->depth();

        sparse_tensor x(arg), self(*this);

        auto it = x.m_data.find(key_type(0));
        if (it != x.m_data.end()) {
            x.m_data.erase(it);
        }

        for (deg_t d=0; d<max_depth; ++d) {
            mul_scal_div(x, Scalar(max_depth-d), d+1);
            *this += self;
        }
        return *this;
    }


    friend std::ostream& operator<<(std::ostream& os, const sparse_tensor& arg)
    {
        os << "{ ";
        for (const auto& val : arg.m_data) {
            os << val.second << arg.m_basis->key_to_string(val.first) << ' ';
        }
        return os << '}';
    }

    dtl::sparse_kv_iterator<sparse_tensor> iterate_kv() const
    {
        return {m_data.begin(), m_data.end()};
    }


    bool operator==(const sparse_tensor& other) const
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

template <typename Scalar>
const Scalar sparse_tensor<Scalar>::zero(0);


template <typename S>
struct algebra_info<sparse_tensor<S>>
{
    using scalar_type = S;

    static constexpr coefficient_type ctype() noexcept
    { return dtl::get_coeff_type(S(0)); }
    static constexpr vector_type vtype() noexcept
    { return vector_type::sparse; }
    static deg_t width(const sparse_tensor<S>& instance) noexcept
    { return instance.width(); }
    static deg_t max_depth(const sparse_tensor<S>& instance) noexcept
    { return instance.depth(); }
    static key_type convert_key(esig::key_type key) noexcept
    { return key; }
};


extern template class sparse_tensor<double>;
extern template class sparse_tensor<float>;

}// namespace algebra
} // namespace esig

#endif//ESIG_PATHS_SRC_ALGEBRA_TMP_SRC_SPARSE_TENSOR_H_
