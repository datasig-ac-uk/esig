//
// Created by sam on 07/03/2022.
//

#ifndef ESIG_PATHS_DENSE_TENSOR_H
#define ESIG_PATHS_DENSE_TENSOR_H


#include <esig/algebra/base.h>
#include "tensor_basis/tensor_basis.h"



#include "dense_kv_iterator.h"

#include <memory>
#include <utility>
#include <vector>
#include <sstream>
#include <memory>
#include <cassert>

namespace esig {
namespace algebra {

template <typename Scalar>
class dense_tensor {
    std::vector<Scalar> m_data;
    std::shared_ptr<const tensor_basis> m_basis;
    deg_t m_degree;

    void check_compatible(const tensor_basis &other_basis) const {
        if (m_basis->width() != other_basis.width()) {
            throw std::invalid_argument("mismatched width");
        }
    }

    void swap(dense_tensor &other) noexcept {
        std::swap(m_data, other.m_data);
        std::swap(m_basis, other.m_basis);
        std::swap(m_degree, other.m_degree);
    }

public:
    using scalar_type = Scalar;
    using basis_type = tensor_basis;
    using reference = Scalar&;
    using const_reference = const Scalar&;
    using const_iterator = dtl::dense_iterator<typename std::vector<Scalar>::const_iterator>;
    static constexpr vector_type vtype = vector_type::dense;

    explicit dense_tensor(deg_t width, deg_t depth)
        : m_basis(new tensor_basis(width, depth)), m_degree(0), m_data() {}

    explicit dense_tensor(std::shared_ptr<const tensor_basis> basis)
        : m_basis(std::move(basis)), m_degree(0), m_data() {}

    dense_tensor(std::shared_ptr<const tensor_basis> basis, deg_t degree)
        : m_basis(std::move(basis)), m_degree(degree), m_data() {}

    dense_tensor(std::shared_ptr<const tensor_basis> basis, deg_t degree, std::initializer_list<Scalar> args)
        : m_basis(std::move(basis)), m_degree(degree), m_data(args) {
        m_data.resize(m_basis->size(static_cast<int>(m_degree)));
    }

    dense_tensor(std::shared_ptr<const tensor_basis> basis, deg_t degree, std::vector<Scalar> &&data)
        : m_basis(std::move(basis)), m_degree(degree), m_data(std::move(data)) {
        m_data.resize(m_basis->size(static_cast<int>(m_degree)));
    }

    dense_tensor(std::shared_ptr<const tensor_basis> basis, const Scalar *begin, const Scalar *end)
        : m_basis(std::move(basis)), m_degree(0), m_data(begin, end) {
        if (!m_data.empty()) {
            resize(m_data.size());
        }
    }

    dense_tensor(std::shared_ptr<const tensor_basis> basis, Scalar val)
        : m_basis(std::move(basis)), m_degree(0), m_data{val} {}

protected:
    void resize(dimn_t new_size) {
        assert(new_size >= m_data.size());
        if (new_size == 0) {
            return;
        }
        auto deg = m_basis->degree(static_cast<key_type>(new_size - 1));
        auto adjusted = m_basis->size(static_cast<int>(deg));
        if (adjusted < new_size) {
            m_degree = deg + 1;
            adjusted = m_basis->size(static_cast<int>(m_degree));
        } else {
            m_degree = deg;
        }
        assert(adjusted >= new_size);
        assert(m_degree <= depth());
        m_data.resize(adjusted);
        assert(m_data.size() == m_basis->size(static_cast<int>(m_degree)));
    }

    void resize_for_key(key_type key) {
        if (key >= m_data.size()) {
            auto deg = m_basis->degree(key);
            if (deg > m_degree) {
                m_degree = deg;
            }
            m_data.resize(m_basis->size(static_cast<int>(m_degree)));
        }
    }

public:
    dimn_t size() const {
        return m_data.size();
    }
    deg_t degree() const {
        return m_degree;
    }
    deg_t width() const {
        return m_basis->width();
    }
    deg_t depth() const {
        return m_basis->depth();
    }
    vector_type storage_type() const noexcept {
        return vector_type::dense;
    }
    coefficient_type coeff_type() const noexcept {
        return dtl::get_coeff_type(Scalar(0));
    }
    const void *start_of_degree(deg_t deg) const {
        return m_data.data() + m_basis->start_of_degree(deg);
    }
    const void *end_of_degree(deg_t deg) const {
        return m_data.data() + m_basis->size(static_cast<int>(deg));
    }

    Scalar *start_of_degree(deg_t deg) {
        return m_data.data() + m_basis->start_of_degree(deg);
    }

    void assign(const std::vector<Scalar> &data) {
        m_data = data;
    }

    template<typename InputIt>
    std::enable_if_t<
        std::is_constructible<
            Scalar,
            typename std::iterator_traits<InputIt>::value_type>::value>
    assign(InputIt begin, InputIt end)
    {
        auto dist = std::distance(begin, end);
        auto max_size = m_basis->size(-1);
        m_data.clear();
        m_data.reserve(std::min(dimn_t(dist), max_size));
        for (auto it = begin; it != end && m_data.size() <= max_size; ++it) {
            m_data.emplace_back(*it);
        }
    }
    template <typename InputIt, typename Traits=std::iterator_traits<InputIt>>
    std::enable_if_t<
        std::is_same<
            std::remove_cv_t<typename Traits::value_type::first_type>,
            key_type
        >::value &&
        std::is_constructible<
            Scalar,
            typename Traits::value_type::second_type
        >::value>
    assign(InputIt begin, InputIt end)
    {
        auto max_size = m_basis->size(-1);
        m_data.clear();
        for (auto it = begin; it != end; ++it) {
            if (it->first < max_size) {
                resize_for_key(it->first);
                m_data[it->first] = Scalar(it->second);
            }
        }
    }


    const std::vector<Scalar>& data() const
    {
        return m_data;
    }

    std::shared_ptr<const tensor_basis> get_basis() const noexcept
    {
        return m_basis;
    }

    const tensor_basis& basis() const
    {
        return *m_basis;
    }

    Scalar& operator[](const key_type& key)
    {
        if (key >= m_data.size()) {
            resize_for_key(static_cast<dimn_t>(key));
        }
        return m_data[key];
    }

    const Scalar& operator[](const key_type& key) const
    {
        return m_data[key];
    }

    const_iterator begin() const noexcept
    {
        return {m_data.begin(), 0};
    }
    const_iterator end() const noexcept
    {
        return {m_data.end(), 0};
    }


private:
    template <typename F>
    dense_tensor binary_operator_impl(const dense_tensor& other, F fn) const
    {
        check_compatible(*other.m_basis);

        auto out_deg = std::min(m_basis->depth(), std::max(m_degree, other.m_degree));
        auto new_size = m_basis->size(out_deg);

        assert(new_size == m_data.size() || new_size == other.m_data.size());

        dense_tensor result(m_basis, out_deg);
        result.m_data.reserve(new_size);

        auto mid = std::min(size(), other.size());
        for (auto i = 0; i < mid; ++i) {
            result.m_data.emplace_back(fn(m_data[i], other.m_data[i]));
        }

        for (auto i=mid; i<m_data.size(); ++i) {
            result.m_data.emplace_back(fn(m_data[i], Scalar(0)));
        }

        for (auto i = mid; i < other.m_data.size(); ++i) {
            result.m_data.emplace_back(fn(Scalar(0), other.m_data[i]));
        }

        return result;
    }

public:
    dense_tensor operator+(const dense_tensor &other) const
    {
        return binary_operator_impl(other, [](const Scalar &a, const Scalar &b) { return a + b; });
    }
    dense_tensor operator-(const dense_tensor &other) const
    {
        return binary_operator_impl(other, [](const Scalar &a, const Scalar &b) { return a - b; });
    }
private:

    template <typename F>
    dense_tensor unary_operator_impl(F fn) const
    {
        dense_tensor result(m_basis, m_degree);
        result.m_data.reserve(m_data.size());
        for (const auto &val : m_data) {
            result.m_data.emplace_back(fn(val));
        }
        return result;
    }

public:
    dense_tensor operator-() const
    {
        return unary_operator_impl([](const Scalar& a) { return -a; });
    }

    dense_tensor operator*(Scalar other) const
    {
        return unary_operator_impl([&](const Scalar& a) { return a*other; });
    }
    dense_tensor operator/(Scalar other) const
    {
        return unary_operator_impl([&](const Scalar &a) { return a / other; });
    }
private:

    template <typename F>
    void inplace_binary_operation_impl(const dense_tensor& other, F fn)
    {

        auto out_deg = std::min(m_basis->depth(), std::max(m_degree, other.m_degree));
        auto new_size = m_basis->start_of_degree(out_deg);


        if (new_size > m_data.size()) {
            resize(new_size);
        }

        auto mid = std::min(size(), other.size());
        for (auto i = 0; i < mid; ++i) {
            fn(m_data[i], other.m_data[i]);
        }
    }

public:

    dense_tensor &operator+=(const dense_tensor &other)
    {
        check_compatible(*other.m_basis);
        inplace_binary_operation_impl(other, [](Scalar& a, const Scalar& b) { return a += b; });
        return *this;
    }
    dense_tensor &operator-=(const dense_tensor &other)
    {
        check_compatible(*other.m_basis);
        inplace_binary_operation_impl(other, [](Scalar &a, const Scalar &b) { return a -= b; });
        return *this;
    }

    dense_tensor& add_scal_prod(const dense_tensor& other, Scalar s)
    {
        inplace_binary_operation_impl(other, [=](Scalar& a, const Scalar& b) { return a += b * s; });
        return *this;
    }

    dense_tensor &sub_scal_prod(const dense_tensor &other, Scalar s)
    {
        inplace_binary_operation_impl(other, [=](Scalar &a, const Scalar &b) { return a -= b * s; });
        return *this;
    }

    dense_tensor &add_scal_div(const dense_tensor &other, Scalar s)
    {
        inplace_binary_operation_impl(other, [=](Scalar &a, const Scalar &b) { return a += b / s; });
        return *this;
    }

    dense_tensor &sub_scal_div(const dense_tensor &other, Scalar s)
    {
        inplace_binary_operation_impl(other, [=](Scalar &a, const Scalar &b) { return a -= b / s; });
        return *this;
    }

    dense_tensor& add_scal_prod(key_type key, Scalar scal)
    {
        assert(key < m_basis->size(m_basis->depth()));
        if (key >= m_data.size()) {
            resize_for_key(key);
        }
        m_data[key] += scal;
        return *this;
    }


    dense_tensor &operator*=(Scalar other)
    {
        for (auto& val : m_data) {
            val *= other;
        }
        return *this;
    }
    dense_tensor &operator/=(Scalar other)
    {
        for (auto& val : m_data) {
            val /= other;
        }
        return *this;
    }
    bool operator==(const dense_tensor &other) const
    {
        if (m_basis->width() != other.m_basis->width()) {
            return false;
        }

        auto mid = std::min(m_data.size(), other.m_data.size());
        for (auto i=0; i<mid; ++i) {
            if (m_data[i] != other.m_data[i]) {
                return false;
            }
        }

        for (auto i=mid; i < m_data.size(); ++i) {
            if (m_data[i] != Scalar(0)) {
                return false;
            }
        }

        for (auto i=mid; i<other.m_data.size(); ++i) {
            if (other.m_data[i] != Scalar(0)) {
                return false;
            }
        }

        return true;

    }
    scalar_t norm_linf() const
    {
        Scalar max_val (0);
        Scalar av;
        for (const auto& val : m_data) {
            if ((av=abs(val)) > max_val) {
                max_val = av;
            }
        }
        return static_cast<scalar_t>(max_val);
    }
    scalar_t norm_l1() const 
    {
        Scalar acc(0);
        for (const auto& val : m_data) {
            acc += abs(val);
        }
        return static_cast<scalar_t>(acc);
    }

private:

    template <typename Op>
    inline static void mul_impl(
            dense_tensor& out_tensor,
            const dense_tensor& lhs,
            const dense_tensor& rhs,
            Op op,
            deg_t max_deg
            ) noexcept
    {
        const auto& powers = dynamic_cast<const tensor_basis&>(*out_tensor.m_basis).powers();
        for (int out_deg = static_cast<int>(max_deg); out_deg >= 0; --out_deg) {
            auto lhs_max_deg = std::min(static_cast<int>(lhs.m_degree), out_deg);
            auto lhs_min_deg = std::max(0, out_deg - static_cast<int>(rhs.m_degree));

            for (int lhs_deg=lhs_max_deg; lhs_deg >= lhs_min_deg; --lhs_deg) {
                int rhs_deg = out_deg - lhs_deg;

                auto* out_ptr = out_tensor.start_of_degree(out_deg);
                const auto* lhs_ptr = reinterpret_cast<const Scalar*>(lhs.start_of_degree(lhs_deg));
                const auto* rhs_ptr = reinterpret_cast<const Scalar*>(rhs.start_of_degree(rhs_deg));

                for (std::ptrdiff_t i=0; i < powers[lhs_deg]; ++i) {
                    for (std::ptrdiff_t j=0; j<powers[rhs_deg]; ++j) {
                        *(out_ptr++) += op(lhs_ptr[i]*rhs_ptr[j]);
                    }
                }
            }
        }
    }


public:
    dense_tensor operator*(const dense_tensor &other) const
    {
        check_compatible(*other.m_basis);

        auto out_deg = std::min(m_basis->depth(), m_degree + other.m_degree);

        dense_tensor tmp(m_basis, out_deg);
        tmp.m_data.resize(m_basis->size(out_deg));

        if (!m_data.empty() && !other.m_data.empty()) {
            mul_impl(tmp, *this, other, [](const Scalar &s) { return s; }, out_deg);
        }
        return tmp;
    }


private:

    template <typename Op>
    inline static void inplace_mul_impl(
            dense_tensor& lhs,
            const dense_tensor& rhs,
            Op op,
            deg_t max_depth
            ) noexcept
    {

        if (rhs.m_data.empty()) {
            lhs.m_data.clear();
            lhs.m_degree = 0;
            return;
        }

        const auto &powers = dynamic_cast<const tensor_basis &>(*lhs.m_basis).powers();

        auto old_lhs_deg = lhs.m_degree;
        if (lhs.m_degree < max_depth) {
            lhs.resize(lhs.m_basis->size(static_cast<int>(max_depth)));
        }

        int offset = (rhs.m_data[0] == Scalar(0)) ? 1 : 0;
        auto* out_ptr_begin = lhs.m_data.data();
        const auto* lhs_ptr_begin = lhs.m_data.data();
        const auto* rhs_ptr_begin = rhs.m_data.data();

        if (max_depth == 0) {
            lhs[0] = lhs[0]*rhs[0];
            return;
        }


        for (int out_deg = static_cast<int>(max_depth); out_deg > 0; --out_deg) {
            int lhs_max_deg = std::min(out_deg - offset, static_cast<int>(old_lhs_deg));
            int lhs_min_deg = std::max(0, out_deg - static_cast<int>(rhs.m_degree));
            bool assign = true;


            auto out_offset = lhs.m_basis->start_of_degree(out_deg);
            for (int lhs_deg = lhs_max_deg; lhs_deg >= lhs_min_deg; --lhs_deg) {
                int rhs_deg = out_deg - lhs_deg;
                auto lhs_offset = lhs.m_basis->start_of_degree(lhs_deg);
                auto rhs_offset = rhs.m_basis->start_of_degree(rhs_deg);
                auto *out_ptr = out_ptr_begin + out_offset;
                const auto* lhs_ptr = lhs_ptr_begin + lhs.m_basis->start_of_degree(static_cast<deg_t>(lhs_deg));
                const auto* rhs_ptr = rhs_ptr_begin + lhs.m_basis->start_of_degree(static_cast<deg_t>(rhs_deg));

                assert(powers[lhs_deg] == (lhs.m_basis->start_of_degree(lhs_deg+1) - lhs.m_basis->start_of_degree(lhs_deg)));
                assert(powers[rhs_deg] == (rhs.m_basis->start_of_degree(rhs_deg+1) - rhs.m_basis->start_of_degree(rhs_deg)));

                for (std::ptrdiff_t i=0; i<powers[lhs_deg]; ++i) {
                    for (std::ptrdiff_t j=0; j<powers[rhs_deg]; ++j) {
                        if (assign) {
                            *(out_ptr++) = op(lhs_ptr[i]*rhs_ptr[j]);
                        } else {
                            *(out_ptr++) += op(lhs_ptr[i]*rhs_ptr[j]);
                        }
                    }
                }
//                assert(out_ptr == lhs_ptr_begin + lhs.m_basis->start_of_degree(out_deg+1));
                assign = false;
            }

        }

        if (!lhs.m_data.empty() && !rhs.m_data.empty()) {
            lhs.m_data[0] = op(lhs.m_data[0]*rhs.m_data[0]);
        } else if (!lhs.m_data.empty()) {
            lhs.m_data[0] = op(Scalar(0));
        }

    }

public:


    dense_tensor &operator*=(const dense_tensor &other)
    {
        check_compatible(*other.m_basis);

        auto out_deg = std::min(m_basis->depth(), m_degree + other.m_degree);

//        dense_tensor tmp(m_basis, out_deg);
//        tmp.m_data.resize(m_basis->size(out_deg));

        inplace_mul_impl(*this, other, [](const Scalar &s) { return s; }, out_deg);
//        this->swap(tmp);
        return *this;
    }

    dense_tensor& mul_scal_prod(const dense_tensor& other, Scalar s, deg_t max_depth)
    {
        auto out_deg = std::min(max_depth, m_degree + other.m_degree);
//        dense_tensor tmp(m_basis, out_deg);
//        tmp.m_data.resize(m_basis->size(out_deg));

        inplace_mul_impl(*this, other, [=](Scalar v) { return v * s; }, out_deg);
//        this->swap(tmp);
        return *this;
    }

    dense_tensor& mul_scal_prod(const dense_tensor& other, Scalar s)
    {
        // assume compatible
        mul_scal_prod(other, s, m_basis->depth());
        return *this;
    }

    dense_tensor &mul_scal_div(const dense_tensor &other, Scalar s, deg_t max_depth)
    {
        auto out_deg = std::min(max_depth, m_degree + other.m_degree);
//        dense_tensor tmp(m_basis, out_deg);
//        tmp.m_data.resize(m_basis->size(out_deg));

        inplace_mul_impl(*this, other, [=](Scalar v) { return v / s; }, out_deg);
        return *this;
    }

    dense_tensor &mul_scal_div(const dense_tensor &other, Scalar s)
    {
        // assume compatible
        mul_scal_div(other, s, m_basis->depth());
        return *this;
    }

    dense_tensor& add_mul(const dense_tensor& lhs, const dense_tensor& rhs)
    {
        auto out_depth = std::min(m_basis->depth(), lhs.degree() + rhs.degree());
        mul_impl(*this, lhs, rhs, [](Scalar s) { return s; }, std::max(m_degree, out_depth));
        return *this;
    }
    dense_tensor& sub_mul(const dense_tensor& lhs, const dense_tensor& rhs)
    {
        auto out_depth = std::min(m_basis->depth(), lhs.degree() + rhs.degree());
        mul_impl(*this, lhs, rhs, [](Scalar s) { return -s; }, std::max(m_degree, out_depth));
        return *this;
    }


    friend dense_tensor exp(const dense_tensor& arg)
    {
        const auto max_depth = arg.m_basis->depth();
        dense_tensor result(arg.m_basis, max_depth);
        result.m_data.resize(arg.m_basis->size(max_depth));
        result.m_data[0] = Scalar(1);

        if (arg.size() == 0) {
            return result;
        }

        for (deg_t d=max_depth; d >= 1; --d) {
            result.mul_scal_div(arg, Scalar(d));
            result.m_data[0] += Scalar(1);
        }
        return result;
    }
    friend dense_tensor log(const dense_tensor& arg)
    {
        const auto max_depth = arg.m_basis->depth();
        dense_tensor x(arg), result(arg.m_basis, max_depth);
        result.m_data.resize(arg.m_basis->size(max_depth));

        x.m_data[0] = Scalar(0);

        for (deg_t d=max_depth; d >= 1; --d) {
            if ((d & 1) == 0) {
                result.m_data[0] -= Scalar(1) / d;
            } else {
                result.m_data[0] += Scalar(1) / d;
            }
            result *= x;
        }
        return result;
    }
    friend dense_tensor inverse(const dense_tensor& arg)
    {
        return dense_tensor(arg.m_basis);
    }
    dense_tensor &fmexp_inplace(const dense_tensor &arg)
    {
        if (arg.m_data.empty()) {
            return *this;
        }

        check_compatible(*arg.m_basis);
        const auto max_depth = m_basis->depth();

        dense_tensor x(arg), self(*this);

        if (m_degree < max_depth) {
            m_data.resize(m_basis->size(max_depth));
            m_degree = max_depth;
        }

        x.m_data[0] = Scalar(0);

        for (deg_t d=max_depth; d >= 1; --d) {
            mul_scal_div(x, Scalar(d), max_depth - d + 1);
            *this += self;
        }

        return *this;
    }


    friend std::ostream& operator<<(std::ostream& os, const dense_tensor& arg)
    {
        os << "{ ";
        key_type k(0);
        for (const auto &val : arg.m_data) {
            if (val != Scalar(0)) {
                os << val << arg.m_basis->key_to_string(k) << ' ';
            }
            ++k;
        }
        os << '}';
        return os;
    }

    dtl::dense_kv_iterator<dense_tensor> iterate_kv() const noexcept
    {
        return {m_data.data(), m_data.data() + m_data.size()};
    }

};


//namespace dtl {
//
//template <typename Scalar>
//struct iterator_helper<dense_kv_iterator<dense_tensor<Scalar>>>
//{
//    using iter_type = dense_kv_iterator<dense_tensor<Scalar>>;
//    static void advance(iter_type& iter)
//    {
//        iter.advance();
//    }
//    static key_type key(const iter_type& iter)
//    {
//        return iter.key();
//    }
//    static const void* value(const iter_type& iter)
//    {
//        return &iter.value();
//    }
//    static bool equals(const iter_type& iter1, const iter_type& iter2)
//    {
//        return iter1.finished();
//    }
//};
//
//} // namespace dtl


template<typename S>
struct algebra_info<dense_tensor<S>>
{
    using algebra_t = dense_tensor<S>;
    using this_key_type = key_type;
    using scalar_type = S;
    using rational_type = S;
    using key_prod_container = boost::container::small_vector_base<std::pair<key_type, int>>;

    static constexpr coefficient_type ctype() noexcept
    { return dtl::get_coeff_type(S(0)); }
    static constexpr vector_type vtype() noexcept
    { return vector_type::dense; }
    static deg_t width(const dense_tensor<S> *instance) noexcept
    { return instance->width(); }
    static deg_t max_depth(const dense_tensor<S> *instance) noexcept
    { return instance->depth(); }

    static deg_t degree(const algebra_t* inst, key_type key) noexcept
    {
        if (inst) {
            return inst->basis().degree(key);
        }
        return 0;
    }
    static deg_t native_degree(const algebra_t* inst, key_type key) noexcept
    {
        return degree(inst, key);
    }


    static key_type convert_key(const algebra_t* instance, esig::key_type key) noexcept
    { return key; }

    static key_type first_key(const dense_tensor<S>* instance) noexcept
    {
        return 0;
    }
    static key_type last_key(const algebra_t* instance) noexcept
    {
        if (instance) {
            return instance->basis().size(-1);
        }
        return 0;
    }

    static algebra_t create_like(const algebra_t& instance)
    {
        // using the initialiser list forces resize to degree.
        return algebra_t(instance.get_basis(), instance.degree(), {S(0)});
    }

    static const key_prod_container& key_product(const algebra_t* instance, key_type k1, key_type k2)
    {
        if (instance) {
            return instance->basis().prod(k1, k2);
        }
        static const boost::container::small_vector<std::pair<key_type, int>, 0> empty;
        return empty;
    }

};

namespace dtl {

template<typename S>
class dense_data_access_implementation<dense_tensor<S>>
: public dense_data_access_interface
{
    const S *m_begin, *m_end;
    key_type m_start_key;

public:
    dense_data_access_implementation(const dense_tensor<S> &alg, key_type start)
    {
        const auto &data = alg.data();
        assert(start < data.size());
        m_begin = data.data() + start;
        m_end = data.data() + data.size();
    }

    dense_data_access_item next() override {
        const S *begin = m_begin, *end = m_end;
        m_begin = nullptr;
        m_end = nullptr;
        return dense_data_access_item(m_start_key, begin, end);
    }
};

}// namespace dtl

extern template class dense_tensor<double>;
extern template class dense_tensor<float>;


} // namespace algebra
} // namespace esig_paths

#endif//ESIG_PATHS_DENSE_TENSOR_H
