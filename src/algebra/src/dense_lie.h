//
// Created by user on 13/03/2022.
//

#ifndef ESIG_PATHS_INCLUDE_ESIG_PATHS_ALGEBRA_LOCAL_DENSE_LIE_H_
#define ESIG_PATHS_INCLUDE_ESIG_PATHS_ALGEBRA_LOCAL_DENSE_LIE_H_

#include <esig/implementation_types.h>
#include <esig/algebra/iteration.h>
#include <esig/algebra/algebra_traits.h>
#include "lie_basis/lie_basis.h"

#include "dense_kv_iterator.h"

#include <memory>
#include <vector>
#include <sstream>


namespace esig {
namespace algebra {


template <typename Scalar>
class dense_lie
{
    std::vector<Scalar> m_data;
    std::shared_ptr<lie_basis> m_basis;
    deg_t m_degree;


    void check_compatible(const lie_basis &other_basis) const
    {
        if (m_basis->width() != other_basis.width()) {
            throw std::invalid_argument("mismatched width");
        }
    }

    void swap(dense_lie &other) noexcept
    {
        std::swap(m_data, other.m_data);
        std::swap(m_basis, other.m_basis);
        std::swap(m_degree, other.m_degree);
    }


public:

    using scalar_type = Scalar;
    using basis_type = lie_basis;


    explicit dense_lie(deg_t width, deg_t depth)
            : m_basis(new lie_basis(width, depth)), m_degree(0), m_data()
    {}

    explicit dense_lie(std::shared_ptr<lie_basis> basis)
            : m_basis(std::move(basis)), m_degree(0), m_data()
    {}

    dense_lie(std::shared_ptr<lie_basis> basis, deg_t degree)
            : m_basis(std::move(basis)), m_degree(degree), m_data()
    {}

    dense_lie(std::shared_ptr<lie_basis> basis, deg_t degree, std::initializer_list<Scalar> args)
            : m_basis(std::move(basis)), m_degree(degree), m_data(args)
    {
        m_data.resize(m_basis->size(static_cast<int>(m_degree)));
    }


    dense_lie(std::shared_ptr<lie_basis> basis, deg_t degree, std::vector<Scalar>&& data)
            : m_basis(std::move(basis)), m_degree(degree), m_data(std::move(data))
    {
        m_data.resize(m_basis->size(static_cast<int>(m_degree)));
    }

    dense_lie(std::shared_ptr<lie_basis> basis, const Scalar* begin, const Scalar* end)
            : m_basis(std::move(basis)), m_degree(0), m_data(begin, end)
    {
        if (!m_data.empty()) {
            m_degree = m_basis->degree(m_data.size());
            m_data.resize(m_basis->size(static_cast<int>(m_degree)));
        }
    }


public:
    dimn_t size() const 
    {
        return m_data.size();
    }
    deg_t degree() const 
    {
        return m_degree;
    }
    deg_t width() const 
    {
        return m_basis->width();
    }
    deg_t depth() const 
    {
        return m_basis->depth();
    }
    vector_type storage_type() const noexcept 
    {
        return vector_type::dense;
    }
    coefficient_type coeff_type() const noexcept 
    {
        return dtl::get_coeff_type(Scalar(0));
    }
    const std::vector<Scalar> &data() const noexcept
    {
        return m_data;
    }
    void clear() noexcept
    {
        m_data.clear();
    }
    const void *start_of_degree(deg_t deg) const
    {
        return m_data.data() + m_basis->start_of_degree(deg);
    }
    const void *end_of_degree(deg_t deg) const 
    {
        return m_data.data() + m_basis->start_of_degree(deg);
    }

    Scalar& operator[](const key_type& key)
    {
        if (key > m_data.size()) {
            m_data.resize(m_basis->size(m_basis->degree(key)));
        }
        return m_data[key-1];
    }

    const Scalar& operator[](const key_type& key) const
    {
        return m_data[key-1];
    }


    dtl::dense_kv_iterator<dense_lie> begin() const noexcept
    {
        return {m_data.data(), m_data.data() + m_data.size(), 1};
    }
    dtl::dense_kv_iterator<dense_lie> end() const noexcept
    {
        return {nullptr, nullptr};
    }

private:
    template<typename F>
    dense_lie binary_operator_impl(const dense_lie &other, F fn) const
    {
        check_compatible(*other.m_basis);

        auto out_deg = std::min(m_basis->depth(), std::max(m_degree, other.m_degree));
        auto new_size = m_basis->size(out_deg);

        assert(new_size == m_data.size() || new_size == other.m_data.size());

        dense_lie result(m_basis, out_deg);
        result.m_data.reserve(new_size);

        auto mid = std::min(size(), other.size());
        for (auto i = 0; i < mid; ++i) {
            result.m_data.emplace_back(fn(m_data[i], other.m_data[i]));
        }

        for (auto i = mid; i < m_data.size(); ++i) {
            result.m_data.emplace_back(fn(m_data[i], Scalar(0)));
        }

        for (auto i = mid; i < other.m_data.size(); ++i) {
            result.m_data.emplace_back(fn(Scalar(0), other.m_data[i]));
        }

        return result;
    }


public:
    dense_lie operator+(const dense_lie &other) const
    {
        return binary_operator_impl(other, [](const Scalar& a, const Scalar& b) { return a + b; });
    }
    dense_lie operator-(const dense_lie &other) const
    {
        return binary_operator_impl(other, [](const Scalar& a, const Scalar& b) {
            return a - b; });
    }
private:
    template<typename F>
    dense_lie unary_operator_impl(F fn) const
    {
        dense_lie result(m_basis, m_degree);
        result.m_data.reserve(m_data.size());
        for (const auto &val : m_data) {
            result.m_data.emplace_back(fn(val));
        }
        return result;
    }

public:
    dense_lie operator-() const
    {
        return unary_operator_impl([](const Scalar& a) { return -a; });
    }
    dense_lie operator*(Scalar other) const
    {
        return unary_operator_impl([=](const Scalar& a) { return a*other; });
    }
    dense_lie operator/(Scalar other) const
    {
        return unary_operator_impl([=](const Scalar& a) { return a/other; });
    }

private:
    template<typename F>
    void inplace_binary_operation_impl(const dense_lie &other, F fn)
    {

        auto out_deg = std::min(m_basis->depth(), std::max(m_degree, other.m_degree));
        auto new_size = m_basis->start_of_degree(out_deg);


        if (new_size > m_data.size()) {
            m_data.resize(new_size);
        }

        auto mid = std::min(size(), other.size());
        for (auto i = 0; i < mid; ++i) {
            fn(m_data[i], other.m_data[i]);
        }
    }

public:

    dense_lie &operator+=(const dense_lie &other)
    {
        check_compatible(*other.m_basis);
        inplace_binary_operation_impl(other, [](Scalar& a, const Scalar& b) { return a += b; });
        return *this;
    }
    dense_lie &operator-=(const dense_lie &other)
    {
        check_compatible(*other.m_basis);
        inplace_binary_operation_impl(other, [](Scalar &a, const Scalar &b) { return a -= b; });
        return *this;
    }
    dense_lie &operator*=(Scalar other)
    {
        for (auto& val : m_data) {
            val *= other;
        }
        return *this;
    }
    dense_lie &operator/=(Scalar other)
    {
        for (auto& val : m_data) {
            val /= other;
        }
        return *this;
    }
    dense_lie &add_scal_prod(const dense_lie &other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](Scalar& a, const Scalar& b) { return a += b*scal; });
        return *this;
    }
    dense_lie &sub_scal_prod(const dense_lie &other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](Scalar& a, const Scalar& b) { return a -= b*scal; });
        return *this;
    }
    dense_lie& add_scal_div(const dense_lie& other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](Scalar& a, const Scalar& b) { return a += b/scal; });
        return *this;
    }
    dense_lie& sub_scal_div(const dense_lie& other, Scalar scal)
    {
        inplace_binary_operation_impl(other, [=](Scalar& a, const Scalar& b) { return a -= b/scal; });
        return *this;
    }

    dense_lie& add_scal_prod(key_type key, Scalar scal)
    {
        auto max = m_basis->size(m_basis->depth());
        assert(key < max);
        if (key >= m_data.size()) {
            m_data.resize(m_basis->size(m_basis->degree(key)));
        }
        m_data[key-1] += scal;
        return *this;
    }
    dense_lie& sub_scal_prod(key_type key, Scalar scal)
    {
        assert(key <= m_basis->size(m_basis->depth()));
        if (key > m_data.size()) {
            m_data.resize(m_basis->size(m_basis->degree(key)));
        }
        m_data[key-1] -= scal;
        return *this;
    }





    bool operator==(const dense_lie &other) const
    {
        if (width() != other.width()) {
            return false;
        }

        auto mid = std::min(m_data.size(), other.m_data.size());
        for (auto i=0; i < mid; ++i) {
            if (m_data[i] != other.m_data[i]) {
                return false;
            }
        }

        for (auto i = mid; i < m_data.size(); ++i) {
            if (m_data[i] != Scalar(0)) {
                return false;
            }
        }

        for (auto i = mid; i < other.m_data.size(); ++i) {
            if (other.m_data[i] != Scalar(0)) {
                return false;
            }
        }

        return true;
    }
    std::string to_string() const 
    {
        std::stringstream ss;
        ss << "{ ";
        key_type k(0);
        for (const auto& val : m_data) {
            if (val != Scalar(0)) {
                ss << val << m_basis->key_to_string(k) << ' ';
            }
        }
        ss << '}';
        return ss.str();
    }
    Scalar norm_linf() const
    {
        Scalar val (0);
        for (const auto& item : m_data) {
            auto ai = abs(item);
            val = (ai > val) ? ai : val;
        }
        return val;
    }
    Scalar norm_l1() const
    {
        Scalar val(0);
        for (const auto& item : m_data) {
            val += abs(item);
        }
        return val;
    }

private:
    template <typename Op>
    inline static void mul_impl(
            dense_lie &out_lie,
            const dense_lie &lhs,
            const dense_lie &rhs,
            Op op) noexcept
    {
        const auto& basis = out_lie.m_basis;
        for (int out_deg = static_cast<int>(out_lie.m_degree); out_deg >= 0; --out_deg) {
            auto lhs_max_deg = std::min(static_cast<int>(lhs.m_degree), out_deg);
            auto lhs_min_deg = std::max(0, out_deg - static_cast<int>(rhs.m_degree));

            for (int lhs_deg = lhs_max_deg; lhs_deg >= lhs_min_deg; --lhs_deg) {
                int rhs_deg = out_deg - lhs_deg;

                for (std::ptrdiff_t i = basis->start_of_degree(lhs_deg); i < basis->size(lhs_deg); ++i) {
                    for (std::ptrdiff_t j = basis->start_of_degree(rhs_deg); j < basis->size(rhs_deg); ++j) {
                        for (const auto& pair : basis->prod(i+1, j+1)) {
                            out_lie.m_data[pair.first-1] += op(static_cast<Scalar>(pair.second)*lhs.m_data[i]*rhs.m_data[j]);
                        }
                    }
                }
            }
        }
    }

public:
    dense_lie operator*(const dense_lie &other) const
    {
        check_compatible(*other.m_basis);

        auto out_deg = std::min(m_basis->depth(), m_degree + other.m_degree);

        dense_lie tmp(m_basis, out_deg);
        tmp.m_data.resize(m_basis->size(out_deg));

        mul_impl(tmp, *this, other, [](const Scalar &s) { return s; });
        return tmp;
    }
    dense_lie &operator*=(const dense_lie &other)
    {
        check_compatible(*other.m_basis);

        auto out_deg = std::min(m_basis->depth(), m_degree + other.m_degree);

        dense_lie tmp(m_basis, out_deg);
        tmp.m_data.resize(m_basis->size(out_deg));

        mul_impl(tmp, *this, other, [](const Scalar &s) { return s; });
        this->swap(tmp);
        return *this;
    }

    dense_lie& add_mul(const dense_lie& lhs, const dense_lie& rhs)
    {
        mul_impl(*this, lhs, rhs, [](Scalar s) { return s; });
        return *this;
    }
    dense_lie& sub_mul(const dense_lie& lhs, const dense_lie& rhs)
    {
        mul_impl(*this, lhs, rhs, [](Scalar s) { return -s; });
        return *this;
    }

    dense_lie& mul_scal_prod(const dense_lie& other, Scalar scal)
    {
        auto out_deg = std::min(m_basis->depth(), m_degree + other.m_degree);
        dense_lie tmp(m_basis, out_deg);
        tmp.m_data.resize(m_basis->size(out_deg));

        mul_impl(tmp, *this, other, [=](Scalar v) { return scal * v; });
        this->swap(tmp);
        return *this;
    }
    dense_lie& mul_scal_div(const dense_lie& other, Scalar scal)
    {
        auto out_deg = std::min(m_basis->depth(), m_degree + other.m_degree);
        dense_lie tmp(m_basis, out_deg);
        tmp.m_data.resize(m_basis->size(out_deg));

        mul_impl(tmp, *this, other, [=](Scalar v) { return v / scal; });
        this->swap(tmp);
        return *this;
    }

    dtl::dense_kv_iterator<dense_lie> iterate_kv() const noexcept
    {
        return {m_data.data(), m_data.data() + m_data.size(), 1};
    }


    friend std::ostream& operator<<(std::ostream& os, const dense_lie& arg)
    {
        os << "{ ";
        key_type k(1);
        for (const auto& val : arg.m_data) {
            if (val != Scalar(0)) {
                os << val << arg.m_basis->key_to_string(k) << ' ';
            }
            ++k;
        }
        return os << '}';
    }

};

template <typename S>
struct algebra_info<dense_lie<S>>
{
    using scalar_type = S;

    static constexpr coefficient_type ctype() noexcept;
    static constexpr vector_type vtype() noexcept;
    static deg_t width(const dense_lie<S>& instance) noexcept;
    static deg_t max_depth(const dense_lie<S>& instance) noexcept;

    using this_key_type = key_type;
    static this_key_type convert_key(esig::key_type key) noexcept;
};

template<typename S>
constexpr coefficient_type algebra_info<dense_lie<S>>::ctype() noexcept
{
    return dtl::get_coeff_type(S(0));
}
template<typename S>
constexpr vector_type algebra_info<dense_lie<S>>::vtype() noexcept
{
    return vector_type::dense;
}
template<typename S>
deg_t algebra_info<dense_lie<S>>::width(const dense_lie<S> &instance) noexcept
{
    return instance.width();
}
template<typename S>
deg_t algebra_info<dense_lie<S>>::max_depth(const dense_lie<S> &instance) noexcept
{
    return instance.depth();
}
template<typename S>
key_type algebra_info<dense_lie<S>>::convert_key(esig::key_type key) noexcept
{
    return key;
}

namespace dtl {

template <typename S>
class dense_data_access_implementation<dense_lie<S>>
    : public dense_data_access_interface
{
    const S* m_begin, *m_end;
    key_type m_start_key;
public:

    dense_data_access_implementation(const dense_lie<S>& alg, key_type start)
    {
        assert(start >= 1);
        const auto& data = alg.data();
        assert(start <= data.size());
        m_begin = data.data() + start - 1;
        m_end = data.data() + data.size();
    }

    dense_data_access_item next() override {
        const S* begin = m_begin, *end = m_end;
        m_begin = nullptr;
        m_end = nullptr;
        return dense_data_access_item(m_start_key, begin, end);
    }
};


} // namespace dtl


extern template class dense_lie<double>;
extern template class dense_lie<float>;

} // namespace algebra
} // namespace esig

#endif//ESIG_PATHS_INCLUDE_ESIG_PATHS_ALGEBRA_LOCAL_DENSE_LIE_H_
