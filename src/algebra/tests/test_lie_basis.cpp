//
// Created by user on 12/05/22.
//

#include <esig/implementation_types.h>
#include <gtest/gtest.h>
#include "lie_basis.h"
#include <libalgebra/libalgebra.h>
#include <libalgebra/rational_coefficients.h>

using esig::deg_t;

namespace {

struct LieBasisFixture : public ::testing::Test
{
    using key_type = esig::key_type;  // same as libalgebra lie key
    static constexpr deg_t width = 5;
    static constexpr deg_t depth = 5;
    esig::algebra::lie_basis esig_lbasis;
    alg::lie_basis<width, depth> la_lbasis;
    using la_lie = alg::lie<alg::coefficients::rational_field, width, depth>;

    LieBasisFixture() : esig_lbasis(width, depth)
    {}

    la_lie esig_prod(key_type k1, key_type k2) const
    {
        la_lie result;
        for (auto& p : esig_lbasis.prod(k1, k2)) {
            result.add_scal_prod(p.first, p.second);
        }
        return result;
    }

    la_lie alg_prod(key_type k1, key_type k2) const
    {
        la_lie l1(k1), l2(k2);
        return l1 * l2;
    }
};


TEST_F(LieBasisFixture, TestProduct)
{
    for (auto k1 : la_lbasis.iterate_keys()) {
        for (auto k2 : la_lbasis.iterate_keys()) {
            EXPECT_EQ(esig_prod(k1, k2), alg_prod(k1, k2));
        }
    }
}


}
