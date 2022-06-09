//
// Created by sam on 12/04/2022.
//

#include <esig/implementation_types.h>
#include <libalgebra/libalgebra.h>
#include <libalgebra/vectors/vectors.h>
#include <libalgebra/coefficients/rational_coefficients.h>

#include "la_esig_equality_helper.h"
#include <random>
#include <utility>
#include <vector>

using namespace esig;
constexpr deg_t WIDTH = 5;
constexpr deg_t DEPTH = 5;

using la_lie = alg::lie<alg::coefficients::rational_field, WIDTH, DEPTH, alg::vectors::sparse_vector>;

class SparseLieTests : public ::testing::Test
{
protected:

    std::mt19937 rng;
    std::uniform_int_distribution<int> dist;

    using basis_t = alg::lie_basis<WIDTH, DEPTH>;
    using scal_t = typename alg::coefficients::rational_field::S;
    using lie = algebra::sparse_lie<scal_t>;


    std::shared_ptr<algebra::lie_basis> basis;

    SparseLieTests() :
        basis(new algebra::lie_basis(WIDTH, DEPTH)),
       rng(std::random_device{}()),
                       dist(-10000, 10000)
    {}

    std::vector<std::pair<const key_type, scal_t>> random_data()
    {
        std::vector<std::pair<const key_type, scal_t>> result;

        auto sz = basis_t::start_of_degree(DEPTH+1);
        result.reserve(sz);

        dimn_t skip = WIDTH;
        for (auto deg=1; deg <= DEPTH; ++deg) {
            std::uniform_int_distribution<key_type> keydist(1, skip-1);
            key_type key = basis_t::start_of_degree(deg);

            while ((key += keydist(rng)) < basis_t::start_of_degree(deg+1)) {
                result.emplace_back(key, scal_t(dist(rng)) / 10000);
            }

            skip *= WIDTH;
        }
        return result;
    }

    scal_t random_scalar()
    {
        return scal_t(dist(rng)) / 10000;
    }

};

TEST_F(SparseLieTests, TestUminus)
{
    auto rd = random_data();

    lie l(basis, rd.begin(), rd.end());
    la_lie el(rd.begin(), rd.end());

    ASSERT_TRUE(LiesEqual(-l, -el));
}

TEST_F(SparseLieTests, TestAdd)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    lie l1(basis, rd1.begin(), rd1.end());
    lie l2(basis, rd2.begin(), rd2.end());

    auto result = l1 + l2;

    la_lie el1(rd1.begin(), rd1.end());
    la_lie el2(rd2.begin(), rd2.end());

    auto expected = el1 + el2;

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(SparseLieTests, TestSub)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    lie l1(basis, rd1.begin(), rd1.end());
    lie l2(basis, rd2.begin(), rd2.end());

    auto result = l1 - l2;

    la_lie el1(rd1.begin(), rd1.end());
    la_lie el2(rd2.begin(), rd2.end());

    auto expected = el1 - el2;

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(SparseLieTests, TestSMul)
{
    auto rd = random_data();
    lie l(basis, rd.begin(), rd.end());
    auto scal = random_scalar();

    auto result = l * scal;

    la_lie el(rd.begin(), rd.end());
    auto expected = el * scal;

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(SparseLieTests, TestSDiv)
{
    auto rd = random_data();
    lie l(basis, rd.begin(), rd.end());
    auto scal = random_scalar();

    auto result = l / scal;

    la_lie el(rd.begin(), rd.end());
    auto expected = el / scal;

    ASSERT_TRUE(LiesEqual(result, expected));
}

TEST_F(SparseLieTests, TestMul)
{
    auto rd1 = random_data();
    auto rd2 = random_data();

    lie l1(basis, rd1.begin(), rd1.end());
    lie l2(basis, rd2.begin(), rd2.end());

    auto result = l1 * l2;

    la_lie el1(rd1.begin(), rd1.end());
    la_lie el2(rd2.begin(), rd2.end());

    auto expected = el1 * el2;

    ASSERT_TRUE(LiesEqual(result, expected));
}
