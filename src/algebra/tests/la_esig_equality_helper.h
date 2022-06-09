//
// Created by sam on 11/04/2022.
//

#ifndef ESIG_LA_ESIG_EQUALITY_HELPER_H
#define ESIG_LA_ESIG_EQUALITY_HELPER_H

#include <libalgebra/libalgebra.h>
#include <dense_lie.h>
#include <sparse_lie.h>
#include <dense_tensor.h>
#include <sparse_tensor.h>


#include <gtest/gtest.h>

template <typename CType, esig::deg_t Width, esig::deg_t Depth>
bool equals_helper(
        const esig::algebra::dense_tensor<typename CType::S>& t1,
        const alg::algebra<alg::free_tensor_basis<Width, Depth>, CType, alg::free_tensor_multiplication<CType>, alg::vectors::dense_vector>& t2)
{
    using base_vec = alg::vectors::dense_vector<alg::free_tensor_basis<Width, Depth>, CType>;

    if (t1.width() != Width) {
        return false;
    }
    if (t1.depth() != Depth) {
        return false;
    }

    const auto& lhs_data = t1.data();
    const auto& rhs_base = alg::vectors::dtl::vector_base_access::convert(t2);

    auto min = std::min(lhs_data.size(), rhs_base.dimension());
    const auto* rhs_data = alg::vectors::dtl::data_access<base_vec>::range_begin(rhs_base);

    for (esig::dimn_t i=0; i < min; ++i) {
        if (lhs_data[i] != rhs_data[i]) {
            return false;
        }
    }

    for (esig::dimn_t i=min; i<lhs_data.size(); ++i) {
        if (lhs_data[i] != CType::zero) {
            return false;
        }
    }

    for (esig::dimn_t i=min; i<rhs_base.dimension(); ++i) {
        if (rhs_data[i] != CType::zero) {
            return false;
        }
    }

    return true;
}


template <typename CType, esig::deg_t Width, esig::deg_t Depth>
bool equals_helper(
        const esig::algebra::dense_lie<typename CType::S>& l1,
        const alg::algebra<alg::lie_basis<Width, Depth>, CType, alg::lie_multiplication<CType>, alg::vectors::dense_vector>& l2)
{
    using base_vec = alg::vectors::dense_vector<alg::lie_basis<Width, Depth>, CType>;

    if (l1.width() != Width) {
        return false;
    }
    if (l1.depth() != Depth) {
        return false;
    }

    const auto& lhs_data = l1.data();
    const auto& rhs_base = alg::vectors::dtl::vector_base_access::convert(l2);

    auto min = std::min(lhs_data.size(), rhs_base.dimension());
    const auto* rhs_data = alg::vectors::dtl::data_access<base_vec>::range_begin(rhs_base);

    for (esig::dimn_t i=0; i < min; ++i) {
        if (lhs_data[i] != rhs_data[i]) {
            return false;
        }
    }

    for (esig::dimn_t i=min; i<lhs_data.size(); ++i) {
        if (lhs_data[i] != CType::zero) {
            return false;
        }
    }

    for (esig::dimn_t i=min; i<rhs_base.dimension(); ++i) {
        if (rhs_data[i] != CType::zero) {
            return false;
        }
    }

    return true;
}

template <typename CType, esig::deg_t Width, esig::deg_t Depth>
bool equals_helper(
        const esig::algebra::sparse_tensor<typename CType::S>& t1,
        const alg::algebra<alg::free_tensor_basis<Width, Depth>, CType, alg::free_tensor_multiplication<CType>, alg::vectors::sparse_vector>& t2
        )
{
    using base_vec = alg::vectors::dense_vector<alg::free_tensor_basis<Width, Depth>, CType>;

    if (t1.width() != Width) {
        return false;
    }
    if (t1.depth() != Depth) {
        return false;
    }


    const auto& rhs_base = alg::vectors::dtl::vector_base_access::convert(t2);

    auto lit = t1.begin();
    auto lend = t1.end();
    auto rit = rhs_base.map_begin();
    auto rend = rhs_base.map_end();

    for (; lit != lend && rit != rend; ++lit, ++rit) {
        // For tensors, the key types don't match, so we have to convert from index to key.
        if (base_vec::basis::index_to_key(lit->first) != rit->first || lit->second != rit->second) {
            return false;
        }
    }
    return lit == lend && rit == rend;
}

template <typename CType, esig::deg_t Width, esig::deg_t Depth>
bool equals_helper(
        const esig::algebra::sparse_lie<typename CType::S>& l1,
        const alg::algebra<alg::lie_basis<Width, Depth>, CType, alg::lie_multiplication<CType>, alg::vectors::sparse_vector>& l2
        )
{
    using base_vec = alg::vectors::dense_vector<alg::lie_basis<Width, Depth>, CType>;

    if (l1.width() != Width) {
        return false;
    }
    if (l1.depth() != Depth) {
        return false;
    }


    const auto& rhs_base = alg::vectors::dtl::vector_base_access::convert(l2);

    auto lit = l1.begin();
    auto lend = l1.end();
    auto rit = rhs_base.map_begin();
    auto rend = rhs_base.map_end();

    for (; lit != lend && rit != rend; ++lit, ++rit) {
        if (lit->first != rit->first || lit->second != rit->second) {
            return false;
        }
    }
    return lit == lend && rit == rend;
}



template<typename CType, esig::deg_t Width, esig::deg_t Depth>
::testing::AssertionResult TensorsEqual(
        const esig::algebra::dense_tensor<typename CType::S>& t1,
        const alg::algebra<alg::free_tensor_basis<Width, Depth>, CType, alg::free_tensor_multiplication<CType>, alg::vectors::dense_vector>& t2)
{
    alg::free_tensor<CType, Width, Depth, alg::vectors::dense_vector> t3(t1.data().data(), t1.data().data() + t1.size());
    if (t3 == t2) {
        return testing::AssertionSuccess();
    } else {
        return testing::AssertionFailure()  << (t2 - t3);
    }
}

template <typename CType, esig::deg_t Width, esig::deg_t Depth>
testing::AssertionResult LiesEqual(
        const esig::algebra::dense_lie<typename CType::S>& l1,
        const alg::algebra<alg::lie_basis<Width, Depth>, CType, alg::lie_multiplication<CType>, alg::vectors::dense_vector>& l2
        )
{
    alg::lie<CType, Width, Depth, alg::vectors::dense_vector> l3(l1.data().data(), l1.data().data() + l1.size());
    if (l2 == l3) {
        return testing::AssertionSuccess();
    } else {
        return testing::AssertionFailure() << (l2 - l3);
    }
}

template <typename CType, esig::deg_t Width, esig::deg_t Depth>
testing::AssertionResult LiesEqual(
        const esig::algebra::sparse_lie<typename CType::S>& l1,
        const alg::algebra<alg::lie_basis<Width, Depth>, CType, alg::lie_multiplication<CType>, alg::vectors::sparse_vector>& l2)
{
    alg::lie<CType, Width, Depth, alg::vectors::sparse_vector> l3(l1.begin(), l1.end());
    if (l2 == l3) {
        return testing::AssertionSuccess();
    } else {
        return testing::AssertionFailure() << (l2 - l3);
    }
}

template <typename CType, esig::deg_t Width, esig::deg_t Depth>
testing::AssertionResult TensorsEqual(
        const esig::algebra::sparse_tensor<typename CType::S>& t1,
        const alg::algebra<alg::free_tensor_basis<Width, Depth>, CType, alg::free_tensor_multiplication<CType>, alg::vectors::sparse_vector>& t2
        )
{
    alg::free_tensor<CType, Width, Depth, alg::vectors::sparse_vector> t3;
    for (const auto& item : t1) {
        t3.add_scal_prod(t3.basis.index_to_key(item.first), item.second);
    }
    if (t2 == t3) {
        return testing::AssertionSuccess();
    } else {
        return testing::AssertionFailure() << (t2 - t3);
    }
}


#endif//ESIG_LA_ESIG_EQUALITY_HELPER_H
