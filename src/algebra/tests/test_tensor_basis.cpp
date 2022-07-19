//
// Created by sam on 19/07/22.
//

#include <esig/implementation_types.h>
#include <libalgebra/libalgebra.h>

#include "../src/tensor_basis/tensor_basis.h"

#include <gtest/gtest.h>

using namespace esig;

class TensorBasisTests : public ::testing::Test
{
protected:

    algebra::tensor_basis basis;
    static constexpr deg_t width = 5;
    static constexpr deg_t depth = 5;

    TensorBasisTests() : basis(width, depth)
    {}


    key_type tensor_key(std::initializer_list<let_t> args) {
        key_type result = 0;
        for (auto arg : args) {
            result *= width;
            result += arg;
        }
        return result;
    }

    key_type concatenate(key_type k1, key_type k2, deg_t deg) {
        return k1*basis.powers()[deg] + k2;
    }

    key_type concat_letter_left(let_t l, key_type k, deg_t deg) {
        return (l - 1)*basis.powers()[deg] + k;
    }


};

TEST_F(TensorBasisTests, test_size) {
    dimn_t size = 1;
    for (deg_t deg = 0; deg <= depth; ++deg) {
        EXPECT_EQ(basis.size(static_cast<int>(deg)), size);
        size *= width;
        size += 1;
    }
}

TEST_F(TensorBasisTests, test_start_of_degree) {
    dimn_t size = 0;
    for (deg_t deg = 0; deg <= depth; ++deg) {
        EXPECT_EQ(basis.start_of_degree(static_cast<int>(deg)), size);
        size *= width;
        size += 1;
    }
}

TEST_F(TensorBasisTests, test_start_of_degree_vs_size) {
    dimn_t diff = 1;
    for (deg_t deg = 0; deg <= depth; ++deg) {
        EXPECT_EQ(basis.size(static_cast<int>(deg))-basis.start_of_degree(deg), diff);
        diff *= width;
    }


}

TEST_F(TensorBasisTests, test_degree) {
    dimn_t size = 1;
    key_type k = 0;
    for (deg_t deg = 0; deg <= depth; ++deg) {

        for (dimn_t i; i<size; ++i, ++k) {
            EXPECT_EQ(basis.degree(k), deg);
        }

        size *= width;
    }
}

TEST_F(TensorBasisTests, test_parents_letters) {
    for (let_t l = 1; l <= width; ++l) {
        EXPECT_EQ(basis.lparent(l), l);
        EXPECT_EQ(basis.rparent(l), 0);
    }
}

TEST_F(TensorBasisTests, test_parents_deg_2) {
    for (let_t l = 1; l<= width; ++l) {
        for (let_t r = 1; r <= width; ++r) {
            auto k = tensor_key({l, r});
            EXPECT_EQ(basis.lparent(k), l);
            EXPECT_EQ(basis.rparent(k), r);
        }
    }
}

TEST_F(TensorBasisTests, test_parents_higher) {
    dimn_t offset = width;
    for (let_t l = 1; l <= width; ++l) {
        key_type k = 0;
        for (deg_t deg; deg < depth; ++deg) {
            for (dimn_t i=0; i<basis.size(static_cast<int>(deg)); ++i, ++k) {
                auto key = concat_letter_left(l, k, deg);
                EXPECT_EQ(basis.lparent(key), l);
                EXPECT_EQ(basis.rparent(key), k);
            }
        }
    }
}
