#ifndef ESIG_ALGEBRA_FALLBACK_OPERATIONS_H_
#define ESIG_ALGEBRA_FALLBACK_OPERATIONS_H_

#include "algebra_fwd.h"
#include "iteration.h"

#include <vector>
#include <utility>

namespace esig {
namespace algebra {
namespace dtl {

template<typename Impl>
using key_value_buffer = std::vector<std::pair<
    typename algebra_info<Impl>::this_key_type,
    typename algebra_info<Impl>::scalar_type>>;

template<typename Impl>
using degree_range_buffer = std::vector<typename key_value_buffer<Impl>::const_iterator>;

bool fallback_equals(const algebra_iterator &lbegin, const algebra_iterator &lend,
                     const algebra_iterator &rbegin, const algebra_iterator &rend) noexcept;

template<typename Interface, typename Impl, typename Fn>
Impl fallback_binary_op(const Impl &lhs, const Interface &rhs, Fn op) {
    Impl result(lhs);
    fallback_inplace_binary_op(result, rhs, op);
    return result;
}

template<typename Interface, typename Impl, typename Fn>
void fallback_inplace_binary_op(Impl &lhs, const Interface &rhs, Fn op) {
    using scalar_type = typename algebra_info<Impl>::scalar_type;

    for (const auto &item : rhs) {
        op(lhs[algebra_info<Impl>::convert_key(&lhs, item.key())], scalars::scalar_cast<scalar_type>(item.value()));
    }
}


template<typename Interface, typename Impl, typename Fn>
void fallback_inplace_multiplication(Impl &result, const Interface &rhs, Fn Op);

template<typename It>
struct multiplication_data {
    It begin;
    It end;
    std::vector<It> degree_ranges;
    deg_t degree;
};

template<typename Impl>
void impl_to_buffer(key_value_buffer<Impl> &buffer,
                    degree_range_buffer<Impl> &degree_ranges,
                    const Impl &arg) {
    using traits = iterator_traits<typename Impl::const_iterator>;
    buffer.reserve(arg.size());
    for (auto it = arg.begin(); it != arg.end(); ++it) {
        buffer.emplace_back(
            traits::key(it),
            traits::value(it));
    }

    std::sort(buffer.begin(), buffer.end(), [](auto lpair, auto rpair) {
      return lpair.first < rpair.first;
    });

    degree_ranges = degree_range_buffer<Impl>(algebra_info<Impl>::max_depth(&arg) + 2, buffer.end());
    deg_t deg = 0;
    for (auto it = buffer.begin(); it != buffer.end() && deg < degree_ranges.size(); ++it) {
        deg_t d = algebra_info<Impl>::native_degree(&arg, it->first);
        while (d > deg) {
            degree_ranges[deg++] = it;
        }
    }
}

template<typename Impl, typename Interface>
void interface_to_buffer(key_value_buffer<Impl> &buffer,
                         degree_range_buffer<Impl> &degree_ranges,
                         const Interface &arg, const Impl *rptr) {
    using info = algebra_info<Impl>;
    using scalar_type = typename info::scalar_type;
    using pair_t = std::pair<typename info::this_key_type, scalar_type>;
    buffer.reserve(arg.size());

    for (auto it = arg.begin(); it != arg.end(); ++it) {
        buffer.emplace_back(info::convert_key(rptr, it->key()), scalars::scalar_cast<scalar_type>(it->value()));
    }

    std::sort(buffer.begin(), buffer.end(),
              [](auto lhs, auto rhs) { return lhs.first < rhs.first; });

    degree_ranges = degree_range_buffer<Impl>(arg.depth() + 2, buffer.end());
    deg_t deg = 0;
    for (auto it = buffer.begin(); it != buffer.end() && deg < degree_ranges.size(); ++it) {
        deg_t d = info::native_degree(rptr, it->first);
        while (d > deg) {
            degree_ranges[deg++] = it;
        }
    }
}
template<typename Algebra, typename LhsIt, typename RhsIt, typename Op>
void fallback_multiplication_impl(Algebra &result,
                                  const multiplication_data<LhsIt> &lhs,
                                  const multiplication_data<RhsIt> &rhs,
                                  Op op) {
    using litraits = iterator_traits<LhsIt>;
    using ritraits = iterator_traits<RhsIt>;
    using info = algebra_info<Algebra>;

    auto max_deg = std::min(info::max_depth(&result), lhs.degree + rhs.degree);

    const auto *ref = &result;
    auto product = [ref](LhsIt lit, RhsIt rit) -> decltype(auto) {
        auto lkey = litraits::key(lit);
        auto rkey = ritraits::key(rit);
        return info::key_product(ref, lkey, rkey);
    };

    for (int out_deg = static_cast<int>(max_deg); out_deg > 0; --out_deg) {
        int lhs_max_deg = std::min(out_deg, static_cast<int>(lhs.degree));
        int lhs_min_deg = std::max(0, out_deg - static_cast<int>(rhs.degree));

        for (int lhs_d = lhs_max_deg; lhs_d >= lhs_min_deg; --lhs_d) {
            auto rhs_d = out_deg - lhs_d;

            for (auto lit = lhs.degree_ranges[lhs_d]; lit != lhs.degree_ranges[lhs_d + 1]; ++lit) {
                for (auto rit = rhs.degree_ranges[rhs_d]; rit != rhs.degree_ranges[rhs_d + 1]; ++rit) {
                    auto val = op(litraits::value(lit) * ritraits::value(rit));
                    for (auto prd : product(lit, rit)) {
                        result[info::convert_key(&result, prd.first)] += prd.second * val;
                    }
                }
            }
        }
    }
}

template<typename Interface, typename Impl, typename Fn>
void fallback_multiplication(Impl &result, const Interface &lhs, const Interface &rhs, Fn op) {
    using info = algebra_info<Impl>;
    using iter = typename key_value_buffer<Impl>::const_iterator;
    key_value_buffer<Impl> lhs_buf, rhs_buf;

    multiplication_data<iter> lhs_data;
    interface_to_buffer(lhs_buf, lhs_data.degree_ranges, lhs, &result);
    lhs_data.begin = lhs_buf.begin();
    lhs_data.end = lhs_buf.end();
    lhs_data.degree = lhs.degree();

    multiplication_data<iter> rhs_data;
    interface_to_buffer<Impl>(rhs_buf, rhs_data.degree_ranges, rhs, &result);
    rhs_data.begin = rhs_buf.begin();
    rhs_data.end = rhs_buf.end();
    rhs_data.degree = rhs.degree();

    multiply_and_add(result, lhs_data, rhs_data, op);
}
template<typename Interface, typename Impl, typename Fn>
void fallback_multiplication(Impl &result, const Impl &lhs, const Interface &rhs, Fn op) {
    using info = algebra_info<Impl>;
    using iter = typename key_value_buffer<Impl>::const_iterator;
    key_value_buffer<Impl> lhs_buf, rhs_buf;

    multiplication_data<iter> lhs_data;
    impl_to_buffer(lhs_buf, lhs_data.degree_ranges, lhs);
    lhs_data.begin = lhs_buf.begin();
    lhs_data.end = lhs_buf.end();
    lhs_data.degree = lhs.degree();

    multiplication_data<iter> rhs_data;
    interface_to_buffer<Impl>(rhs_buf, rhs_data.degree_ranges, rhs, &result);
    rhs_data.begin = rhs_buf.begin();
    rhs_data.end = rhs_buf.end();
    rhs_data.degree = rhs.degree();

    multiply_and_add(result, lhs_data, rhs_data, op);
}
template<typename Interface, typename Impl, typename Fn>
void fallback_multiplication(Impl &result, const Interface &lhs, const Impl &rhs, Fn op) {
    using info = algebra_info<Impl>;
    using iter = typename key_value_buffer<Impl>::const_iterator;
    key_value_buffer<Impl> lhs_buf, rhs_buf;

    multiplication_data<iter> lhs_data;
    interface_to_buffer(lhs_buf, lhs_data.degree_ranges, lhs, &result);
    lhs_data.begin = lhs_buf.begin();
    lhs_data.end = lhs_buf.end();
    lhs_data.degree = lhs.degree();

    multiplication_data<iter> rhs_data;
    impl_to_buffer<Impl>(rhs_buf, rhs_data.degree_ranges, rhs);
    rhs_data.begin = rhs_buf.begin();
    rhs_data.end = rhs_buf.end();
    rhs_data.degree = rhs.degree();

    multiply_and_add(result, lhs_data, rhs_data, op);
}

template<typename Interface, typename Impl, typename Fn>
void fallback_inplace_multiplication(Impl &result, const Interface &rhs, Fn op) {
    auto tmp = algebra_info<Impl>::create_like(result);
    fallback_multiplication(tmp, result, rhs, op);
    result = tmp;
}




template<typename Algebra, typename Interface, typename Fn>
Algebra fallback_multiply(const Algebra &lhs, const Interface &rhs, Fn op) {
    auto result = algebra_info<Algebra>::create_like(lhs);
    fallback_multiplication(result, lhs, rhs, op);
    return result;
}

template<typename Algebra, typename LhsIt, typename RhsIt>
struct multiplication_dispatcher {
    template<typename Op>
    static void dispatch(Algebra &result,
                         const multiplication_data<LhsIt> &lhs,
                         const multiplication_data<RhsIt> &rhs,
                         Op op) {
        fallback_multiplication_impl(result, lhs, rhs, op);
    }
};
template<typename Algebra, typename LhsIt, typename RhsIt, typename Op>
void multiply_and_add(Algebra &result,
                      dtl::multiplication_data<LhsIt> lhs,
                      dtl::multiplication_data<RhsIt> rhs,
                      Op op) {
    /*
     * The dispatcher can be specialised by specific implementations to
     * use a native multiplication operation where possible. Otherwise it
     * uses the fallback multiplication defined in algebra_traits.h
     */
    dtl::multiplication_dispatcher<Algebra, LhsIt, RhsIt>::dispatch(result, lhs, rhs, op);
}



}
}// namespace algebra
}// namespace esig

#endif// ESIG_ALGEBRA_FALLBACK_OPERATIONS_H_
